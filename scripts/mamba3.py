from math import prod, sqrt
from typing import NamedTuple
from tinygrad import Device, Tensor, nn

def shift_right(x: Tensor) -> Tensor:
    return Tensor.cat(Tensor.zeros_like(x[:, :1]), x[:, :-1], dim=1)

def segsum(x: Tensor) -> Tensor:
    T = x.shape[-1]
    x = x.unsqueeze(-1).expand(*x.shape, T)
    rows = Tensor.arange(T, device=x.device).reshape(T, 1)
    cols = Tensor.arange(T, device=x.device).reshape(1, T)
    lower = rows > cols
    lower_eq = rows >= cols
    x = lower.where(x, 0)
    x_segsum = x.cumsum(-2)
    return lower_eq.where(x_segsum, -float("inf"))

def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    x1 = x[..., 0::2] # even
    x2 = x[..., 1::2] # odd

    cos_a = angles.cos()
    sin_a = angles.sin()

    x_rot_even = cos_a * x1 - sin_a * x2
    x_rot_odd = sin_a * x1 + cos_a * x2

    return Tensor.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)

def ssd_mimo(x, A, B, C, chunk_size, initial_states=None):
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    x, A, B, C = [
        Tensor.rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = Tensor.rearrange(A, "b c l h -> b h c l")
    A_cumsum = A.cumsum(-1)

    L = segsum(A).exp()
    Y_diag = Tensor.einsum("bclhnq, bcshnr, bhcls, bcshpr -> bclhpq", C, B, L, x)

    decay_states = (A_cumsum[:, :, :, -1:] - A_cumsum).exp()
    states = Tensor.einsum("bclhnr, bhcl, bclhpr -> bchpn", B, decay_states, x)

    if initial_states is None:
        initial_states = Tensor.zeros_like(states[:, :1])
    states = Tensor.cat(initial_states, states, dim=1)
    decay_chunk = segsum(A_cumsum[:, :, :, -1].pad((1, 0))).exp()
    new_states = Tensor.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = A_cumsum.exp()
    Y_off = Tensor.einsum("bclhnr, bchpn, bhcl -> bclhpr", C, states, state_decay_out)
    Y = Tensor.rearrange(Y_diag + Y_off, "b c l h p r -> b (c l) h p r")

    return Y, final_state

def ssd(x, A, B, C, chunk_size, initial_states=None):
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    x, A, B, C = [
        Tensor.rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = Tensor.rearrange(A, "b c l h -> b h c l")
    A_cumsum = A.cumsum(-1)

    L = segsum(A).exp()
    Y_diag = Tensor.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    decay_states = Tensor.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = Tensor.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    if initial_states is None:
        initial_states = Tensor.zeros_like(states[:, :1])
    states = Tensor.cat(initial_states, states, dim=1)
    decay_chunk = segsum(A_cumsum[:, :, :, -1].pad((1, 0))).exp()
    new_states = Tensor.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = Tensor.exp(A_cumsum)
    Y_off = Tensor.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    Y = Tensor.rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state

class InferenceCache(NamedTuple):
    ssm_state: Tensor
    prev_Bx: Tensor
    cum_angle: Tensor

    @staticmethod
    def alloc(batch_size: int, n_heads: int, head_dim: int, state_dim: int):
        return InferenceCache(
            ssm_state=Tensor.zeros(batch_size, n_heads, head_dim, state_dim),
            prev_Bx=Tensor.zeros(batch_size, n_heads, head_dim, state_dim),
            cum_angle=Tensor.zeros(batch_size, 1, state_dim // 2),
        )

class Mamba3Block:
    def __init__(self, dim: int, state_dim: int, hidden_mult: int, n_heads: int, mimo_rank: int | None, chunk_size: int):
        bound = 1/sqrt(dim)
        if not mimo_rank:
            self.mimo_rank = 1
        else:
            self.mimo_rank = mimo_rank

        self.dim = dim
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_mult * self.dim
        self.bc_dim = state_dim * self.mimo_rank
        assert self.hidden_dim % self.n_heads == 0, f"hidden_dim (dim * {hidden_mult}) must be divisble by n_heads"
        self.head_dim = self.hidden_dim // self.n_heads
        self.chunk_size = chunk_size

        # z: hidden_dim ; gate for output (SiLU activation)
        # x: hidden_dim ; SSM input value
        # B: state_dim*self.mimo_rank ; SSM input projection (d_state for SISO, d_state*R for MIMO)
        # C: state_dim*self.mimo_rank ; SSM output projection
        # dt: n_heads ; step size Δ (one per head)
        # λ: n_heads ; trapezoidal interpolation parameter (one per head)
        # θ: state_dim // 2 ; rotation angles for data-dependent RoPE
        dim_in_proj = 2*self.hidden_dim + 2*self.bc_dim + 2*n_heads + state_dim // 2
        self.in_proj = nn.Linear(dim, dim_in_proj, bias=False)

        self.A_log = Tensor.ones(self.n_heads, requires_grad=True)
        self.D = Tensor.ones(self.n_heads, requires_grad=True)
        self.dt_bias = Tensor.ones(self.n_heads, requires_grad=True)

        self.B_bias = Tensor.ones(n_heads, state_dim, self.mimo_rank, requires_grad=True)
        self.C_bias = Tensor.ones(n_heads, state_dim, self.mimo_rank, requires_grad=True)

        self.B_norm = nn.RMSNorm(self.bc_dim, eps=1e-5)
        self.C_norm = nn.RMSNorm(self.bc_dim, eps=1e-5)

        if mimo_rank is not None:
            self.mimo_x_proj = Tensor.ones(n_heads, self.head_dim, self.mimo_rank, requires_grad=True)
            self.mimo_z_proj = Tensor.ones(n_heads, self.head_dim, self.mimo_rank, requires_grad=True)
            self.mimo_down = Tensor.ones(n_heads, self.head_dim, self.mimo_rank, requires_grad=True) / self.mimo_rank

        self.out_norm = nn.RMSNorm(self.hidden_dim, eps=1e-5)
        self.out_proj = nn.Linear(self.hidden_dim, self.dim, bias=False)

    def init_state(self, batch_size: int):
        return InferenceCache.alloc(batch_size, self.n_heads, self.head_dim, self.state_dim)

    def __call__(self, u: Tensor, h: InferenceCache | None = None):
        if h is not None:
            return self.step(u, h)

        batch, seqlen, dim = u.shape

        A = -self.A_log.exp() # always <0

        proj = self.in_proj(u)
        z, x, B, C, dt, lam, theta = proj.split(
            [
                self.hidden_dim, self.hidden_dim,
                self.bc_dim, self.bc_dim,
                self.n_heads, self.n_heads,
                self.state_dim // 2,
            ], dim=-1
        )
        
        dt = (dt + self.dt_bias).softplus()
        lam = lam.sigmoid()

        B = self.B_norm(B)
        C = self.C_norm(C)

        raw_angles = dt.unsqueeze(-1) * Tensor.rearrange(theta, "b l n -> b l 1 n")
        angles = -raw_angles.cumsum(1)

        dA = dt * Tensor.rearrange(A, "h -> 1 1 h")
        alpha = dA.exp()
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt

        x = Tensor.rearrange(x, "b l (h p) -> b l h p", p=self.head_dim)

        if self.mimo_rank != 1:
            R = self.mimo_rank

            B = Tensor.rearrange(B, "b l (n r) -> b l n r", r=R)
            C = Tensor.rearrange(C, "b l (n r) -> b l n r", r=R)
            B = Tensor.rearrange(B, "b l n r -> b l 1 n r") + self.B_bias
            C = Tensor.rearrange(C, "b l n r -> b l 1 n r") + self.C_bias

            B = Tensor.rearrange(B, "b l h n r -> b l h r n")
            C = Tensor.rearrange(C, "b l h n r -> b l h r n")
            B = apply_rope(B, angles.unsqueeze(3))
            C = apply_rope(C, angles.unsqueeze(3))
            B = Tensor.rearrange(B, "b l h r n -> b l h n r")
            C = Tensor.rearrange(C, "b l h r n -> b l h n r")

            x_mimo = x.unsqueeze(-1) * self.mimo_x_proj
            y_gamma, state_gamma = ssd_mimo(
                x_mimo * Tensor.rearrange(gamma, "b l h -> b l h 1 1"),
                dA, B, C, self.chunk_size
            )

            B_prev = shift_right(B)
            x_mimo_prev = shift_right(x_mimo)

            y_beta, state_beta = ssd_mimo(
                x_mimo_prev * Tensor.rearrange(beta, "b l h -> b l h 1 1"),
                dA, B_prev, C, self.chunk_size
            )

            y = y_gamma + y_beta
            ssm_state = state_gamma + state_beta

            y = y + (x * self.D.unsqueeze(-1)).unsqueeze(-1)

            z_heads = Tensor.rearrange(z, "b l (h p) -> b l h p", p=self.head_dim)
            z_mimo = z_heads.unsqueeze(-1) * self.mimo_z_proj
            y = y * z_mimo.silu()

            y = (y * self.mimo_down).sum(-1)
            y = Tensor.rearrange(y, "b l h p -> b l (h p)")
            y = self.out_proj(y)

            last_Bx = Tensor.einsum(
                "bhnr, bhpr -> bhpn",
                B[:, -1], x_mimo[:, -1],
            )
        else:
            B = Tensor.rearrange(B, "b l n -> b l 1 n") + self.B_bias.squeeze(-1)
            C = Tensor.rearrange(C, "b l n -> b l 1 n") + self.C_bias.squeeze(-1)
            B = apply_rope(B, angles)
            C = apply_rope(C, angles)

            y_gamma, state_gamma = ssd(
                x * gamma.unsqueeze(-1), dA, B, C, self.chunk_size,
            )

            B_prev = shift_right(B)
            x_prev = shift_right(x)

            y_beta, state_beta = ssd(
                x_prev * beta.unsqueeze(-1), dA, B_prev, C, self.chunk_size,
            )

            y = y_gamma + y_beta
            ssm_state = state_gamma + state_beta

            y = y + x * self.D.unsqueeze(-1)

            y = Tensor.rearrange(y, "b l h p -> b l (h p)")
            y = y * z.silu()
            y = self.out_proj(y)

            last_Bx = Tensor.einsum(
                "bhn, bhp -> bhpn",
                B[:, -1], x[:, -1],
            )

        last_angle = angles[:, -1:]
        h_new = InferenceCache(
            ssm_state=ssm_state,
            prev_Bx=last_Bx,
            cum_angle=last_angle.squeeze(1)
        )
        return y, h_new
    
    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        A = -self.A_log.exp()

        proj = self.in_proj(u.squeeze(1))
        z, x, B, C, dt, lam, theta = proj.split(
            [
                self.hidden_dim, self.hidden_dim,
                self.bc_dim, self.bc_dim,
                self.n_heads, self.n_heads,
                self.state_dim // 2,
            ], dim=-1
        )

        dt = (dt + self.dt_bias).softplus()
        lam = lam.sigmoid()

        B = self.B_norm(B)
        C = self.C_norm(C)

        raw_angle = (
            dt.unsqueeze(-1)
            * theta.unsqueeze(1)
        ) 
        new_cum_angle = h.cum_angle - raw_angle

        dA = dt * A
        alpha = dA.exp()
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt

        x = Tensor.rearrange(x, "b (h p) -> b h p", p=self.head_dim)

        if self.mimo_rank != 1:
            R = self.mimo_rank

            B = Tensor.rearrange(B, "b (n r) -> b n r", r=R)
            C = Tensor.rearrange(C, "b (n r) -> b n r", r=R)
            B = B.unsqueeze(1) + self.B_bias
            C = C.unsqueeze(1) + self.C_bias

            B = Tensor.rearrange(B, "b h n r -> b h r n")
            C = Tensor.rearrange(C, "b h n r -> b h r n")
            B = apply_rope(B, new_cum_angle.unsqueeze(2))  # broadcast over rank
            C = apply_rope(C, new_cum_angle.unsqueeze(2))
            B = Tensor.rearrange(B, "b h r n -> b h n r")
            C = Tensor.rearrange(C, "b h r n -> b h n r")

            # ── Expand x to rank R ──
            x_mimo = x.unsqueeze(-1) * self.mimo_x_proj  # (batch, nheads, headdim, R)

            # ── MIMO state update: B @ X^T contracts rank R ──
            BX = Tensor.einsum("bhnr, bhpr -> bhpn", B, x_mimo)

            new_ssm_state = (
                h.ssm_state * Tensor.rearrange(alpha, "b h -> b h 1 1")
                + h.prev_Bx * Tensor.rearrange(beta, "b h -> b h 1 1")
                + BX * Tensor.rearrange(gamma, "b h -> b h 1 1")
            )

            # ── Output: H^T @ C → (headdim, R) per head ──
            y = Tensor.einsum("bhpn, bhnr -> bhpr", new_ssm_state, C)

            # ── Skip connection in rank-R space ──
            y = y + (x * Tensor.rearrange(self.D, "h -> 1 h 1")).unsqueeze(-1)

            # ── Gate in rank-R space ──
            z_heads = Tensor.rearrange(z, "b (h p) -> b h p", p=self.head_dim)
            z_mimo = z_heads.unsqueeze(-1) * self.mimo_z_proj  # (batch, nheads, P, R)
            y = y * z_mimo.silu()

            # ── Down-project rank ──
            y = (y * self.mimo_down).sum(-1)
            y = Tensor.rearrange(y, "b h p -> b (h p)")
            y = self.out_proj(y)

            h_new = InferenceCache(
                ssm_state=new_ssm_state,
                prev_Bx=BX,
                cum_angle=new_cum_angle,
            )
            return y.unsqueeze(1), h_new

        else:
            # ── SISO path ──
            B = B.unsqueeze(1) + self.B_bias.squeeze(-1)
            C = C.unsqueeze(1) + self.C_bias.squeeze(-1)
            B = apply_rope(B, new_cum_angle)
            C = apply_rope(C, new_cum_angle)

            Bx = Tensor.einsum("bhn, bhp -> bhpn", B, x)

            new_ssm_state = (
                h.ssm_state * Tensor.rearrange(alpha, "b h -> b h 1 1")
                + h.prev_Bx * Tensor.rearrange(beta, "b h -> b h 1 1")
                + Bx * Tensor.rearrange(gamma, "b h -> b h 1 1")
            )

            y = Tensor.einsum("bhpn, bhn -> bhp", new_ssm_state, C)
            y = y + Tensor.rearrange(self.D, "h -> h 1") * x

            y = Tensor.rearrange(y, "b h p -> b (h p)")
            y = y * z.silu()
            y = self.out_proj(y)

            h_new = InferenceCache(
                ssm_state=new_ssm_state,
                prev_Bx=Bx,
                cum_angle=new_cum_angle,
            )
            return y.unsqueeze(1), h_new

class SwiGLUBlock:
    def __init__(self, dim: int, hidden_dim: int | None = None):
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim * 4
        self.w = nn.Linear(dim, self.hidden_dim, bias=True)
        self.v = nn.Linear(dim, self.hidden_dim, bias=True)
        self.out = nn.Linear(self.hidden_dim, dim, bias=True)

    def __call__(self, x: Tensor):
        return self.out(self.w(x).silu() * self.v(x))

class Mamba3:
    def __init__(self, dim: int, blocks: int, state_dim: int | None = None, hidden_mult: int = 2, n_heads: int = 8, mlp_mult: int = 4, mimo_rank: int = 1, chunk_size: int = 64):
        self.dim = dim
        self.state_dim = state_dim if state_dim is not None else dim
        self.mimo_rank = mimo_rank
        self.layers = []
        self.norm = nn.RMSNorm(dim)

        for _ in range(blocks):
            self.layers.append({
                "mixer_norm": nn.RMSNorm(dim),
                "mixer": Mamba3Block(dim, self.state_dim, hidden_mult, n_heads, mimo_rank, chunk_size),
                "mlp_norm": nn.RMSNorm(dim),
                "mlp": SwiGLUBlock(dim, dim * mlp_mult),
            })

    def init_state(self, batch_size: int):
        return [layer["mixer"].init_state(batch_size) for layer in self.layers]

    def __call__(self, x: Tensor, state=None):
        use_step = state is not None and x.shape[1] == 1
        if not use_step:
            state = [None] * len(self.layers)

        h = x
        new_state = []
        for i, layer in enumerate(self.layers):
            y, layer_state = layer["mixer"](layer["mixer_norm"](h), state[i])
            h = h + y
            h = h + layer["mlp"](layer["mlp_norm"](h))
            new_state.append(layer_state)
        return self.norm(h), new_state

class Mamba3LMHeadModel:
    def __init__(self, dim: int, blocks: int, vocab_size: int, state_dim: int | None = None, hidden_mult: int = 2, n_heads: int = 8, mlp_mult: int = 4, mimo_rank: int = 1, chunk_size: int = 64):
        self.embed = nn.Embedding(vocab_size, dim)
        self.backbone = Mamba3(dim, blocks, state_dim=state_dim, hidden_mult=hidden_mult, n_heads=n_heads, mlp_mult=mlp_mult, mimo_rank=mimo_rank, chunk_size=chunk_size)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def init_state(self, batch_size: int):
        return self.backbone.init_state(batch_size)

    def __call__(self, input_ids: Tensor, state=None):
        x = self.embed(input_ids)
        x, state = self.backbone(x, state)
        return self.lm_head(x), state

if __name__ == "__main__":
    print("Device:", Device.DEFAULT)

    m = Mamba3LMHeadModel(dim=10, blocks=2, vocab_size=32, state_dim=20, n_heads=2, mimo_rank=2, chunk_size=2)
    params = nn.state.get_parameters(m)
    print("n_params:", sum(prod(p.shape) for p in params))

    seq = Tensor([[1, 2, 3, 4]])
    ys, _ = m(seq)
    print("ys:", ys.numpy())
    print("ys.shape:", ys.shape)
