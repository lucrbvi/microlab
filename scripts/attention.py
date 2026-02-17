import math

from tinygrad import Device, Tensor

# Attention
def attn(q: Tensor, k: Tensor, v: Tensor, dim: int):
    core = (q @ k.T)/math.sqrt(dim)
    return core.softmax() @ v

# Multi Head Attention
def mha(x: Tensor, heads: int, dim: int, key_dim: int):
    out = []
    wo = Tensor.randn(heads * key_dim, dim)

    for h in range(heads):
        wq = Tensor.randn(dim, key_dim)
        wk = Tensor.randn(dim, key_dim)
        wv = Tensor.randn(dim, key_dim)

        out.append(attn(x @ wq, x @ wk, x @ wv, key_dim))

    merged = out[0]
    for t in out[1:]:
        merged = merged.cat(t, dim=1)

    return merged @ wo

# Positional Encoding
def pe(pos: int, i: int, dim: int):
    if i % 2 == 0:
        return math.sin(pos/(10000**(2*math.floor(i/2)/dim)))
    else:
        return math.cos(pos/(10000**(2*math.floor(i/2)/dim)))

# Apply PE for a whole tensor
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

if __name__ == "__main__":
    print(Device.DEFAULT)
    tokens = Tensor([[1.2, 1.5, 1.2], [0.7, 2.0, 0.7], [1.2, 1.5, 1.2]])
    m = mha(tokens, 5, 3, 2)
    print(f"MHA: \n{m.numpy()}\n")
    print(f"MHA + PE: \n{pet(m).numpy()}\n")
