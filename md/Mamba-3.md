[See the code](../scripts/mamba3.py)

### Recap

Mamba-3 is the latest evolution of the Mamba family of architectures. Mamba can be seen as a reccurent neural network with a hidden state. All architectures of the Mamba family are State Space Models (SSMs).

A Mamba model is formed of multiple Mamba blocks.

### State-Space Models (SSMs), Mamba-1 and Mamba-2

SSMs are defined with a continuous-time linear dynamics, this will be discretized later:

$$\dot{h}(t)=A(t)h(t)+B(t)x(t)$$
$$y(t)=C(t)^Th(t)$$

$h(t)∈ℝ^N$ is the hidden state at time-step $t$, $x(t)∈ℝ$ the input and $A(t)∈ℝ^{N×N}, (B(t), C(t))∈ℝ^N$. $y(t)$ is the output of the model.

$A(t)$ is refered as the *state-transition* and $B(t)x(t)$ the *state-input*.

Here is how Mamba-1 and Mamba-2 discretized this system with $Δ_t$ as the step size:

$$h_t=e^{Δ_tA_t}h_{t-1}+Δ_tB_tx_t$$
$$y_t=C^T_th_t$$

This can be simplified with parametrization (replacing some values with learnable parameters) as:

$$h_t=\alpha_th_{t-1}+\gamma_tB_tx_t$$
$$y_t=C^T_th_t$$

### Mamba-3

Mamba-3 is an evolution of Mamba-2 with three innovations:
- "Exponential-trapezoidal" discretization
- Complex-valued state space
- Multi-Input, Multi-Output (MIMO)

These innovations aims to close the performance gap between Mamba-2 and Self-Attention in Transformers.

#### "Exponential-trapezoidal" discretization

The discretization used in Mamba-1 and Mamba-2 is only using the current token; Mamba-3 improves it by injecting the previous token, it gives more capabilites to the model.

$$h_t=\alpha_th_{t-1}+\beta_{t}B_{t-1}x_{t-1}+\gamma_tB_tx_t$$

This new discretization method can be seen as an approximation of a width-2 convolution since the previous and current tokens are processed.

#### Complex-valued state space

A known limitation of Mamba-2 is the state transition $\alpha_t∈[0;1]$. Even though it is performant, it cannot be negative which limit the abilitiy of the SSM to apply advanced transformations on the hidden-state.

To adress this limit, the authors of Mamba-3 introduce complex values to rotate the hidden-state, which improve the ability to transform the information. However, these complex values can be approximied with a single block matrix.

$$θ_t∈ℝ^{N/2}$$
$$R_t:=Block(\sum_{i=1}^{N/2}Δ_tθ_t[i])∈ℝ^{N×N}$$

If you include this proposition with the "exponential-trapezoidal" discretization you obtain:

$$h_t=\alpha_th_{t-1}+\beta_t(\prod^{t-1}_{i=0}R^T_i)B_{t-1}x_{t-1}+\gamma_t(\prod^{t}_{i=0}R^T_i)B_tx_t$$
$$y_t=[(\prod^{t}_{i=0}R^T_i)C_t]^Th_t$$

#### Multi-Input, Multi-Output (MIMO)

MIMO modify $b_t, c_t, x_t$ and $y_t$ (all vectors) by "stacking" them $R$ times. This give us multiple matrices.

$$h_t^{(j)} ← \alpha_th_{t-1}^{(j)}+Δ_tB_t^{(j)}x_t^{(j)}$$
$$h_t = \sum^{R-1}_{j=0}h_t^{(j)}$$
$$y_t^{(i)}←(C_t^{(i)})^Th_t$$
$$y_t=\sum^{R}_{i=0}y_t^{(i)}$$

This give the model more capabilities while not being more memory hungry and not slowing down training nor inference.

#### Other architecture modifications

The overall architecture of Mamba-3 is based one from Llama 3 alternating Mamba-3 and SwiGLU blocks with pre-norm. Here are some improvements compared to the Mamba-2 block:
- $BC$ Normalization: RMS normalizations are added following the $B$ and $C$ projection, inspired from QK Normalization used in modern Transformers (like Llama 3).
- $BC$ Biases