from math import prod
from tinygrad import Device, Tensor, nn

class Mamba3Block:
    def __init__(self, dim: int, state_dim: int | None = None, mimo_rank: int = 1):
        self.dim = dim
        self.state_dim = state_dim if state_dim is not None else dim
        self.mimo_rank = mimo_rank

        if self.state_dim % 2 != 0:
            raise ValueError("Mamba-3's real-valued complex state requires an even dim")
        if self.mimo_rank <= 0:
            raise ValueError("mimo_rank must be positive")

        self.half_state_dim = self.state_dim // 2

        self.a_proj = nn.Linear(dim, 1, bias=True) # A_t < 0
        self.dt_proj = nn.Linear(dim, 1, bias=True) # Delta_t > 0
        self.lam_proj = nn.Linear(dim, 1, bias=True) # lambda_t in [0,1]

        self.theta_proj = nn.Linear(dim, self.half_state_dim, bias=True)

        proj_dim = self.state_dim * self.mimo_rank
        self.B_proj = nn.Linear(dim, proj_dim, bias=True)
        self.C_proj = nn.Linear(dim, proj_dim, bias=True)

        self.norm_eps = 1e-5
        self.B_norm = nn.RMSNorm(self.state_dim, eps=self.norm_eps)
        self.C_norm = nn.RMSNorm(self.state_dim, eps=self.norm_eps)
        self.B_bias = Tensor.zeros(self.state_dim, self.mimo_rank, requires_grad=True)
        self.C_bias = Tensor.zeros(self.state_dim, self.mimo_rank, requires_grad=True)
        self.x_scale = (Tensor.ones(dim, self.mimo_rank, requires_grad=True) / (self.dim ** 0.5))
        self.y_scale = (Tensor.ones(dim, self.mimo_rank, requires_grad=True) / ((self.state_dim * self.mimo_rank) ** 0.5))

    def rotate(self, v: Tensor, phase: Tensor) -> Tensor:
        if len(v.shape) == 1:
            vp = v.reshape(self.half_state_dim, 2)
            real = vp[:, 0] * phase.cos() + vp[:, 1] * phase.sin()
            imag = -vp[:, 0] * phase.sin() + vp[:, 1] * phase.cos()
            return Tensor.stack(real, imag, dim=-1).reshape(self.state_dim)

        if len(v.shape) != 2 or v.shape[0] != self.state_dim:
            raise ValueError(f"expected state tensor with shape ({self.state_dim}, k), got {v.shape}")

        vp = v.reshape(self.half_state_dim, 2, v.shape[1])
        cos_t = phase.cos().reshape(self.half_state_dim, 1)
        sin_t = phase.sin().reshape(self.half_state_dim, 1)
        real = vp[:, 0] * cos_t + vp[:, 1] * sin_t
        imag = -vp[:, 0] * sin_t + vp[:, 1] * cos_t
        return Tensor.stack(real, imag, dim=1).reshape(self.state_dim, v.shape[1])

    def norm_state(self, norm: nn.RMSNorm, x: Tensor) -> Tensor:
        return norm(x.transpose()).transpose()

    def expand_input(self, x: Tensor) -> Tensor:
        return x.reshape(self.dim, 1) * self.x_scale

    def project(self, x: Tensor):
        A_t = -self.a_proj(x).softplus()
        Delta_t = self.dt_proj(x).softplus()
        lambda_t = self.lam_proj(x).sigmoid()

        theta_t = self.theta_proj(x)

        B_t = self.norm_state(self.B_norm, self.B_proj(x).reshape(self.state_dim, self.mimo_rank)) + self.B_bias
        C_t = self.norm_state(self.C_norm, self.C_proj(x).reshape(self.state_dim, self.mimo_rank)) + self.C_bias

        alpha_t = (Delta_t * A_t).exp()
        beta_t = (1.0 - lambda_t) * Delta_t * alpha_t
        gamma_t = lambda_t * Delta_t

        return Delta_t, alpha_t, beta_t, gamma_t, theta_t, B_t, C_t

    def init_state(self):
        return (
            Tensor.zeros(self.state_dim, self.dim),   # hidden_t
            None,                                     # B_{t-1}
            None,                                     # x_{t-1}
            Tensor.zeros(self.half_state_dim),        # phase_{t-1}
        )

    def __call__(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor | None, Tensor | None, Tensor] | None = None,
    ):
        if len(x.shape) != 1 or x.shape[0] != self.dim:
            raise ValueError(f"expected x.shape == ({self.dim},), got {x.shape}")

        hidden, prev_B, prev_x, prev_phase = self.init_state() if state is None else state
        Delta_t, alpha_t, beta_t, gamma_t, theta_t, B_t, C_t = self.project(x)
        x_t = self.expand_input(x)

        phase_t = prev_phase + Delta_t.reshape(1) * theta_t

        if prev_B is None or prev_x is None:
            prev_term = Tensor.zeros(self.state_dim, self.dim)
        else:
            prev_term = beta_t.reshape(1, 1) * (self.rotate(prev_B, prev_phase) @ prev_x.transpose())

        current_term = gamma_t.reshape(1, 1) * (self.rotate(B_t, phase_t) @ x_t.transpose())
        hidden_t = alpha_t.reshape(1, 1) * hidden + prev_term + current_term

        score_t = (self.rotate(C_t, phase_t).transpose() @ hidden_t).transpose()
        y_t = x + (score_t * self.y_scale).sum(axis=-1)

        new_state = (hidden_t, B_t, x_t, phase_t)
        return y_t, new_state

class SwiGLUBlock:
    def __init__(self, dim: int, hidden_dim: int | None = None):
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim * 4
        self.norm = nn.RMSNorm(dim)
        self.w = nn.Linear(dim, self.hidden_dim, bias=True)
        self.v = nn.Linear(dim, self.hidden_dim, bias=True)
        self.out = nn.Linear(self.hidden_dim, dim, bias=True)

    def __call__(self, x: Tensor):
        h = self.norm(x)
        return x + self.out(self.w(h).silu() * self.v(h))

class Mamba3:
    def __init__(self, dim: int, blocks: int, state_dim: int | None = None, mimo_rank: int = 1):
        self.dim = dim
        self.state_dim = state_dim if state_dim is not None else dim
        self.mimo_rank = mimo_rank
        self.blocks = []
        for _ in range(blocks):
            self.blocks.extend((Mamba3Block(dim, self.state_dim, mimo_rank=mimo_rank), SwiGLUBlock(dim)))

    def init_state(self):
        return [block.init_state() for block in self.blocks if isinstance(block, Mamba3Block)]

    def __call__(self, x: Tensor, state=None):
        if state is None:
            state = self.init_state()

        single_token = len(x.shape) == 1
        tokens = [x] if single_token else [x[i] for i in range(x.shape[0])]

        outputs = []
        cur_state = state

        for token in tokens:
            h = token
            new_state = []
            state_idx = 0

            for block in self.blocks:
                if isinstance(block, Mamba3Block):
                    h, next_block_state = block(h, cur_state[state_idx])
                    new_state.append(next_block_state)
                    state_idx += 1
                else:
                    h = block(h)

            cur_state = new_state
            outputs.append(h)

        if single_token:
            return outputs[0], cur_state

        return Tensor.cat(*[o.reshape(1, self.dim) for o in outputs], dim=0), cur_state

if __name__ == "__main__":
    print("Device:", Device.DEFAULT)

    m = Mamba3(dim=10, blocks=10, state_dim=20, mimo_rank=2)
    params = nn.state.get_parameters(m)
    n_params = sum(prod(p.shape) for p in params)
    print(n_params)

    token = Tensor.ones(1, 10)
    y1, state = m(token, state=None)
    y2, state = m(token, state=state)
    print("y1:", y1.numpy())
    print("y2:", y2.numpy())
