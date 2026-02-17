import math

from tinygrad import Device, Tensor, TinyJit, nn

# Attention
def attn(q: Tensor, k: Tensor, v: Tensor, dim: int):
    core = (q @ k.T) / math.sqrt(dim)
    return core.softmax() @ v

# Single head
class attn_head:
    def __init__(self, dim: int, key_dim: int):
        bound = 1 / math.sqrt(dim)
        self.key_dim = key_dim

        self.wq = Tensor.uniform(dim, key_dim, low=-bound, high=bound)
        self.wk = Tensor.uniform(dim, key_dim, low=-bound, high=bound)
        self.wv = Tensor.uniform(dim, key_dim, low=-bound, high=bound)

    def __call__(self, x: Tensor):
        return attn(x @ self.wq, x @ self.wk, x @ self.wv, self.key_dim)

# Muli-Head Attention
class mha:
    def __init__(self, dim: int, key_dim: int, heads: int):
        bound = 1 / math.sqrt(dim)

        self.dim = dim
        self.key_dim = key_dim
        self.nheads = heads
        self.heads = [attn_head(dim, key_dim) for _ in range(heads)]
        self.wo = Tensor.uniform(heads * key_dim, dim, low=-bound, high=bound)

    def __call__(self, x: Tensor):
        out = []
        for i in range(self.nheads):
            out.append(self.heads[i](x))
        return Tensor.cat(*out, dim=1) @ self.wo

# Positional Encoding
def pe(pos: int, i: int, dim: int):
    if i % 2 == 0:
        return math.sin(pos / (10000 ** (2 * math.floor(i / 2) / dim)))
    else:
        return math.cos(pos / (10000 ** (2 * math.floor(i / 2) / dim)))

# Apply PE for a whole tensor
@TinyJit
def pet(x: Tensor):
    if len(x.shape) != 2:
        raise ValueError(f"Expected x to have 2 dimensions, found {x.shape}")

    seq_len, dim = x.shape
    new_x = []

    for pos in range(seq_len):
        row = []
        for i in range(dim):
            row.append(pe(pos, i, dim))
        new_x.append(row)

    pe_tensor = Tensor(new_x, device=x.device, dtype=x.dtype)
    return x + pe_tensor

# The Transformer from "Attention is all you need"
class Transformer:
    def __init__(
        self,
        dim: int = 512,
        heads: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 6,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        vocab_size: int = 32000, # English->French
    ):
        self.dim, self.eps = dim, eps
        key_dim, hidden, bound = dim // heads, dim * ffn_mult, 1 / math.sqrt(dim)
        self.enc = [
            self._new_block(dim, key_dim, heads, hidden, bound)
            for _ in range(enc_layers)
        ]
        self.dec = [
            self._new_block(dim, key_dim, heads, hidden, bound, cross=True)
            for _ in range(dec_layers)
        ]
        self.last = nn.Linear(dim, vocab_size)

    def _new_block(
        self,
        dim: int,
        key_dim: int,
        heads: int,
        hidden: int,
        bound: float,
        cross: bool = False,
    ):
        block = {
            "self": mha(dim, key_dim, heads),
            "cross": mha(dim, key_dim, heads) if cross else None,
            "w1": Tensor.uniform(dim, hidden, low=-bound, high=bound),
            "b1": Tensor.zeros(hidden),
            "w2": Tensor.uniform(hidden, dim, low=-bound, high=bound),
            "b2": Tensor.zeros(dim),
            "ln1_gamma": Tensor.ones(dim),
            "ln1_beta": Tensor.zeros(dim),
            "ln2_gamma": Tensor.ones(dim),
            "ln2_beta": Tensor.zeros(dim),
        }
        if cross:
            block["lnc_gamma"], block["lnc_beta"] = Tensor.ones(dim), Tensor.zeros(dim)
        return block

    def _layer_norm(self, x: Tensor, gamma: Tensor, beta: Tensor):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) * (x - mean)).mean(axis=-1, keepdim=True)
        xhat = (x - mean) / (var + self.eps).sqrt()
        return xhat * gamma.reshape(1, self.dim) + beta.reshape(1, self.dim)

    def ffn(self, x: Tensor, block: dict):
        h = x @ block["w1"] + block["b1"].reshape(1, -1)
        return h.relu() @ block["w2"] + block["b2"].reshape(1, -1)

    def _run_stack(self, x: Tensor, stack: list, enc_out: Tensor | None = None):
        for block in stack:
            x = self._layer_norm(
                x + block["self"](x), block["ln1_gamma"], block["ln1_beta"]
            )
            if block["cross"] is not None and enc_out is not None:
                x = self._layer_norm(
                    x + block["cross"](x + enc_out),
                    block["lnc_gamma"],
                    block["lnc_beta"],
                )
            x = self._layer_norm(
                x + self.ffn(x, block), block["ln2_gamma"], block["ln2_beta"]
            )
        return x

    def encode(self, src: Tensor):
        return self._run_stack(pet(src), self.enc)

    def decode(self, tgt: Tensor, enc_out: Tensor):
        return self._run_stack(pet(tgt), self.dec, enc_out)

    @TinyJit
    def __call__(self, src: Tensor, tgt: Tensor | None = None):
        if tgt is None:
            tgt = src
        enc_out = self.encode(src)
        out = self.last(self.decode(tgt, enc_out))
        return out.softmax()

if __name__ == "__main__":
    print(Device.DEFAULT)
    tokens = Tensor([[1.2, 1.5, 1.2], [0.7, 2.0, 0.7], [1.2, 1.5, 1.2]])
    m = mha(3, 1, 4)(tokens)
    print(f"MHA: \n{m.numpy()}\n")
    print(f"MHA + PE: \n{pet(m).numpy()}\n\n")

    trans = Transformer()
    print(trans(Tensor.randn((3, 512))).numpy())
