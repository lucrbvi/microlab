import math
import urllib.request
import csv
import os
import time
import numpy as np
from collections import Counter

from tinygrad import Device, Tensor, TinyJit, dtypes, nn


# Attention
def attn(q: Tensor, k: Tensor, v: Tensor, dim: int, mask: Tensor | None = None):
    core = (q @ k.T) / math.sqrt(dim)
    if mask is not None:
        core = core + mask
    return core.softmax() @ v


# Single head
class attn_head:
    def __init__(self, dim: int, key_dim: int):
        bound = 1 / math.sqrt(dim)
        self.key_dim = key_dim

        self.wq = Tensor.uniform(
            dim, key_dim, low=-bound, high=bound, requires_grad=True
        )
        self.wk = Tensor.uniform(
            dim, key_dim, low=-bound, high=bound, requires_grad=True
        )
        self.wv = Tensor.uniform(
            dim, key_dim, low=-bound, high=bound, requires_grad=True
        )

    def __call__(
        self,
        x: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        mask: Tensor | None = None,
    ):
        k = k if k is not None else x
        v = v if v is not None else x
        return attn(x @ self.wq, k @ self.wk, v @ self.wv, self.key_dim, mask)


# Muli-Head Attention
class mha:
    def __init__(self, dim: int, key_dim: int, heads: int):
        bound = 1 / math.sqrt(dim)

        self.dim = dim
        self.key_dim = key_dim
        self.nheads = heads
        self.heads = [attn_head(dim, key_dim) for _ in range(heads)]
        self.wo = Tensor.uniform(
            heads * key_dim, dim, low=-bound, high=bound, requires_grad=True
        )

    def __call__(
        self,
        x: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        mask: Tensor | None = None,
    ):
        out = []
        for i in range(self.nheads):
            out.append(self.heads[i](x, k, v, mask))
        return Tensor.cat(*out, dim=1) @ self.wo


# Positional Encoding
def pe(pos: int, i: int, dim: int):
    if i % 2 == 0:
        return math.sin(pos / (10000 ** (2 * math.floor(i / 2) / dim)))
    else:
        return math.cos(pos / (10000 ** (2 * math.floor(i / 2) / dim)))


def pet(x: Tensor):
    seq_len, dim = x.shape
    pos = Tensor.arange(seq_len, device=x.device).reshape(-1, 1).float()
    i = Tensor.arange(dim, device=x.device).reshape(1, -1).float()

    div_term = 10000 ** (2 * (i // 2) / dim)
    pe = Tensor.where(i % 2 == 0, (pos / div_term).sin(), (pos / div_term).cos())
    return x + pe


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
        vocab_size: int = 32000,  # English->French
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
        self.embed = nn.Embedding(vocab_size, dim)

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
            "w1": Tensor.uniform(
                dim, hidden, low=-bound, high=bound, requires_grad=True
            ),
            "b1": Tensor.zeros(hidden, requires_grad=True),
            "w2": Tensor.uniform(
                hidden, dim, low=-bound, high=bound, requires_grad=True
            ),
            "b2": Tensor.zeros(dim, requires_grad=True),
            "ln1_gamma": Tensor.ones(dim, requires_grad=True),
            "ln1_beta": Tensor.zeros(dim, requires_grad=True),
            "ln2_gamma": Tensor.ones(dim, requires_grad=True),
            "ln2_beta": Tensor.zeros(dim, requires_grad=True),
        }
        if cross:
            block["lnc_gamma"], block["lnc_beta"] = (
                Tensor.ones(dim, requires_grad=True),
                Tensor.zeros(dim, requires_grad=True),
            )
        return block

    def _layer_norm(self, x: Tensor, gamma: Tensor, beta: Tensor):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) * (x - mean)).mean(axis=-1, keepdim=True)
        xhat = (x - mean) / (var + self.eps).sqrt()
        return xhat * gamma.reshape(1, self.dim) + beta.reshape(1, self.dim)

    def ffn(self, x: Tensor, block: dict):
        h = x @ block["w1"] + block["b1"].reshape(1, -1)
        return h.relu() @ block["w2"] + block["b2"].reshape(1, -1)

    def _run_stack(
        self,
        x: Tensor,
        stack: list,
        enc_out: Tensor | None = None,
        causal: bool = False,
    ):
        mask = None
        if causal:
            seq_len = x.shape[0]
            mask = (Tensor.ones(seq_len, seq_len, device=x.device) * -1e9).triu(1)
        for block in stack:
            x = self._layer_norm(
                x + block["self"](x, mask=mask), block["ln1_gamma"], block["ln1_beta"]
            )
            if block["cross"] is not None and enc_out is not None:
                x = self._layer_norm(
                    x + block["cross"](x, enc_out, enc_out),
                    block["lnc_gamma"],
                    block["lnc_beta"],
                )
            x = self._layer_norm(
                x + self.ffn(x, block), block["ln2_gamma"], block["ln2_beta"]
            )
        return x

    def encode(self, src: Tensor):
        return self._run_stack(pet(self.embed(src)), self.enc)

    def decode(self, tgt: Tensor, enc_out: Tensor):
        return self._run_stack(pet(self.embed(tgt)), self.dec, enc_out, causal=True)

    def __call__(self, src: Tensor, tgt: Tensor | None = None):
        if tgt is None:
            tgt = src
        enc_out = self.encode(src)
        out = self.last(self.decode(tgt, enc_out))
        return out


def tokenize(s):
    return s.lower().split()


def build_train_step(model: Transformer, optim, pad_id: int):
    @TinyJit
    def train_step(src_ids: Tensor, dec_ids: Tensor, y_ids: Tensor):
        optim.zero_grad()
        ignore = y_ids == pad_id
        target = Tensor.where(ignore, -1, y_ids)
        loss = model(src_ids, dec_ids).sparse_categorical_crossentropy(
            target, ignore_index=-1
        )
        loss.backward()
        optim.step()
        return loss

    return train_step


def build_vocab(pairs: list[tuple[str, str]], max_vocab: int = 8000):
    counter = Counter()
    for en, fr in pairs:
        counter.update(tokenize(en))
        counter.update(tokenize(fr))

    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab += [w for w, _ in counter.most_common(max_vocab - len(vocab))]
    stoi = {w: i for i, w in enumerate(vocab)}
    return stoi


def encode_text(text: str, stoi: dict[str, int], max_len: int = 32):
    ids = [stoi["<bos>"]]
    ids += [stoi.get(w, stoi["<unk>"]) for w in tokenize(text)][: max_len - 2]
    ids += [stoi["<eos>"]]
    ids += [stoi["<pad>"]] * (max_len - len(ids))
    return ids


# train the transformer on a subset of WMT 2014 FR-EN
def train_transformer(
    model: Transformer | None = None,
    optim=None,
    epochs: int = 35,
    batch_size: int = 16,
    max_vocab: int = 40000,
    max_len: int = 50,
):
    pairs = []  # store tuples (en, fr)
    if not os.path.exists("data.csv"):
        resp = urllib.request.urlopen(
            "https://storage.googleapis.com/kagglesdsdata/datasets/4484220/7685004/wmt14_translate_fr-en_test.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260218%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260218T115921Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9553c2604fed50d2294798a3f3b66b3f00feb744e7679022ef21ccf746ea3b39a2d0e6f5e0fc418116e9400edcbe6e5be05db7e6d6f9ace605df5ef5d1ae06b4c719762e093e10a81aa56f5e0180c3caae91e48ae0bfc27213208bafce7a8bffa65f147be147ef9289a2e0b44aa2a17c9d702c9794cfe3be44dc5b9e1be6d363c2bfb168ea8c80a5bce678088a788164096fd1ce11eb2035716050a399913eeffa7376fc075608cc4ff51f787f3e3eb58db813402659823782918c9a6457ebbb6d4026455be87a9547df82af3f6b6ec5e09faf02531b6212c73505ab69d125b6ec11c5838155044e990bc34e34408f86127be1bc4e986a63253745fae2146d6c"
        )
        with open("data.csv", "wb") as file:
            file.write(resp.read())
    with open("data.csv", mode="r") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if len(row) < 2:
                continue
            en, fr = row[0].strip(), row[1].strip()
            if not en or not fr or en == "en" or fr == "fr":
                continue
            pairs.append((en, fr))

    if len(pairs) < 2:
        raise ValueError("Not enough sentence pairs in data.csv")

    stoi = build_vocab(pairs, max_vocab=max_vocab)
    print(len(stoi))
    if model is None:
        model = Transformer(
            dim=30,
            heads=4,
            enc_layers=3,
            dec_layers=3,
            ffn_mult=3,
            vocab_size=len(stoi),
        )
    if optim is None:
        optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-3)

    Tensor.training = True
    train_step = build_train_step(model, optim, stoi["<pad>"])

    warmup = min(2, max(1, (len(pairs) - 1) // batch_size))
    for step_idx in range(warmup):
        en, fr = pairs[(step_idx * batch_size) % (len(pairs) - 1)]
        src_ids = Tensor(
            encode_text(en, stoi, max_len=max_len),
            dtype=dtypes.int32,
            device=Device.DEFAULT,
        )
        tgt_ids = encode_text(fr, stoi, max_len=max_len)
        dec_ids = Tensor(tgt_ids[:-1], dtype=dtypes.int32, device=Device.DEFAULT)
        y_ids = Tensor(tgt_ids[1:], dtype=dtypes.int32, device=Device.DEFAULT)
        train_step(src_ids, dec_ids, y_ids).realize()

    total_steps = epochs * max(1, (len(pairs) - 1) // batch_size)
    for step_idx in range(total_steps):
        t0 = time.perf_counter()
        en, fr = pairs[(step_idx * batch_size) % (len(pairs) - 1)]
        src_ids = Tensor(
            encode_text(en, stoi, max_len=max_len),
            dtype=dtypes.int32,
            device=Device.DEFAULT,
        )
        tgt_ids = encode_text(fr, stoi, max_len=max_len)
        dec_ids = Tensor(tgt_ids[:-1], dtype=dtypes.int32, device=Device.DEFAULT)
        y_ids = Tensor(tgt_ids[1:], dtype=dtypes.int32, device=Device.DEFAULT)
        loss = train_step(src_ids, dec_ids, y_ids).realize()
        print(
            f"step: {step_idx}, loss: {loss.item()}, dt: {time.perf_counter() - t0:.3f}s"
        )

    model.stoi = stoi
    model.itos = ["<unk>"] * len(stoi)
    for w, idx in stoi.items():
        model.itos[idx] = w
    model.max_len = max_len
    return model


if __name__ == "__main__":
    print(Device.DEFAULT)
    tokens = Tensor([[1.2, 1.5, 1.2], [0.7, 2.0, 0.7], [1.2, 1.5, 1.2]])
    m = mha(3, 1, 4)(tokens)
    print(f"MHA: \n{m.numpy()}\n")
    print(f"MHA + PE: \n{pet(m).numpy()}\n\n")

    trans = train_transformer()

    Tensor.training = False
    en_text = "I love you"
    temperature = 0.8
    top_k = 20
    min_decode_len = 3
    src_ids = Tensor(
        encode_text(en_text, trans.stoi, max_len=trans.max_len),
        dtype=dtypes.int32,
        device=Device.DEFAULT,
    )
    dec_tokens = [trans.stoi["<bos>"]]
    for _ in range(trans.max_len - 1):
        dec_ids = Tensor(dec_tokens, dtype=dtypes.int32, device=Device.DEFAULT)
        logits = trans(src_ids, dec_ids).realize()
        last_logits = logits[-1].numpy().astype(np.float64)
        if len(dec_tokens) <= min_decode_len:
            last_logits[trans.stoi["<eos>"]] = -1e9
            last_logits[trans.stoi["<unk>"]] = -1e9
        k = min(top_k, last_logits.shape[0])
        top_idx = np.argpartition(last_logits, -k)[-k:]
        top_logits = last_logits[top_idx] / temperature
        top_logits -= top_logits.max()
        probs = np.exp(top_logits)
        probs /= probs.sum()
        next_id = int(np.random.choice(top_idx, p=probs))
        dec_tokens.append(next_id)
        if next_id == trans.stoi["<eos>"]:
            break

    out_words = []
    for tok in dec_tokens[1:]:
        if tok == trans.stoi["<eos>"] or tok == trans.stoi["<pad>"]:
            break
        if tok < len(trans.itos):
            w = trans.itos[tok]
            if w not in ("<bos>", "<eos>", "<pad>"):
                out_words.append(w)
    print(f"EN: {en_text}")
    print(f"FR: {' '.join(out_words) if out_words else '<empty>'}")
