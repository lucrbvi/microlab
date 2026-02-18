[See the code](../scripts/attention.py)
### Recap
Attention is a simple mechanism to calculate similarities between internal tokens in transformers.
It's designed to be parallel, it means it's really fast to train and run in GPUs and other accelerators.
$$∀(Q,K,V)∈ℝ^{N×d_k}, Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
`d_k` is the dimension of the keys in the equation and `N` is the number of tokens. A single token is a single vector.
$$∀(W^Q,W^K,W^V)∈ℝ^{d_{model}×d_k}, W^O∈ℝ^{d_{model}d_k×d_{model}}$$
$$Multi Head Attention = Concat(head_1,...,head_n)W^O$$
$$head_n = Attention(XW^Q,XW^K,XW^V)$$
In the design of Multi-Head Attention, each keys is in a lower dimensional space for efficiency.
### Positional Encoding
Since the attention mechanism is not recurrent, the model cannot "know" what's the position of each tokens we need to introduce a way to differentiate tokens by their positions.
$$PE_{(pos, 2i)}=sin(pos/10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)}=cos(pos/10000^{2i/d_{model}})$$
In this equation `pos` in the position of the token inside the sequence and `i` is the dimension.
These equations are calculated every `dimension` times.
It means there is two dimensions calculated `i` times .

This positional encoding is added to each tokens:
$$∀x∈ℝ^{d_{model}}$$
$$\text{PE}(pos, j) = \begin{cases} \sin\left(pos/10000^{2\lfloor j/2\rfloor / d_{\text{model}}}\right) & \text{if } j \text{ is even} \\[10pt] \cos\left(pos/10000^{2\lfloor j/2\rfloor / d_{\text{model}}}\right) & \text{if } j \text{ is odd} \end{cases}$$
$$\displaystyle\sum_{i=0}^nx_i'=x_i+PE(x_i)$$
Positional encoding is essential for transformers because they treat everything in parallel. So in the phrase "Where is John? John is here" the model cannot distinguish the word "John" from the other apparition of the same word, they have the same value.
The positional encoding add a signal based on the position.

## Conclusion
This paper introduced the fundamental concepts of transformers, even if those concepts were known for a long time, they made a clear design on how to use it effectively.