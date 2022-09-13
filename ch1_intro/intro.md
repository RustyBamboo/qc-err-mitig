# Introduction


## Pure States

Suppose we have a two-state quantum system: a qubit. We label the two discrete states as $\ket{0} = [1 \; 0]^T$ and $\ket{1} = [0 \; 1]^T$, which denote the _computational_ basis. The general state of the qubit is then described as a _superposition_ of these two states:

$$
\ket{\psi} = \alpha \ket{0} + \beta \ket{1} = [\alpha \; \beta]^T, \quad |\alpha|^2 + |\beta|^2 = 1
$$

where $\alpha, \beta \in C$. This in stark contrast to classical computers where the state of a bit is fixed to either $0$ or $1$. It is often convenient to work in a different basis, such as the _plus-minus_ basis:

$$
\ket{0} = \frac{\ket{+} + \ket{-}}{\sqrt{2}}, \quad \ket{1} = \frac{\ket{+} - \ket{-}}{\sqrt{2}}.
$$

In order to transform the basis, quantum gates can be applied to the qubit to manipulate the input state to a desired state.
For example, the change-of-basis matrix that maps the _computational_ basis to the _plus-minus_ basis is the Hadamard gate:

$$
H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}.
$$

Assuming the initial state of a qubit is $\ket{0}$, then preforming the Hadamard gate yields: 

$$ 
H\ket{0} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) = \ket{+}.
$$

### Several qubits

To describe the collective space of $n$-qubits in terms of the individual qubits, the _tensor product_ (_Kronecker product_) is used: $\otimes$. In vector-notation, the number of elements is $2^n$. For example, if all $n$-qubits are in the $\ket{0}$ state, then their collective state is:

$$
\ket{0} \otimes \ket{0} ... \otimes \ket{0} = \ket{0}^{\otimes n} = \ket{00... 0} = [1 \; 0 \; ... \; 0]^T
$$

```{list-table} Equivalent representations of the same example state $\ket{3}$ with 3-qubits
:header-rows: 1
:name: representation

* - Type
  - Representation
* - Tensor Product 
  - $\ket{0} \otimes \ket{1} \otimes \ket{1}$ 
* - Base-2 State
  - $\ket{011}$
* - Base-10 State
  - $\ket{3}$ 
* - Column Vector
  - $[0 \; 0 \; 0 \; 1 \; 0 \; 0 \; 0 \; 0]^T$
```

In general, an arbitrary $n$-qubit state will be a superposition of the basis formed by the tensor product:

$$
\ket{\psi} = a_0 \ket{0} + a_1\ket{1} + ... + a_n \ket{n},
$$

where $\ket{k}$ for $k < n$ is base-10 equivalent of the base-2 representation (e.g. for 5-qubits, $\ket{7} = \ket{0} \otimes \ket{0} \otimes \ket{1} \otimes \ket{1} \otimes \ket{1} = \ket{00111})$. A list summarizing the various representations is shown in {numref}`representation`.

It is important to note that within the $n$-qubit space, there are states that cannot be written in terms of tensor products of individual qubit states. These states are known as _entangled_ states. Suppose, for example, we have a 3-qubit system in the state

$$
\ket{\psi} = \frac{1}{\sqrt{2}}(\ket{000} + \ket{111}).
$$

Then there is no tensor product of three individual qubits, $\ket{\phi_a} = \alpha\ket{0} + \beta\ket{0}$ and $\ket{\phi_b} = \gamma \ket{0} + \delta \ket{1}$ and $\ket{\phi_c} = \epsilon \ket{0} + \zeta \ket{1}$  that will give $\ket{\psi}$. These entangled states are pivotal for the theoretical exponential algorithmic speed-up.


## Mixed States

In the previous section, we made an assumption that we have absolute knowledge of the state of our qubits. This enables us to write a column vector that completely describes our state. However, sometimes we may encounter scenarios where there is an underlying classical uncertainty in the actual quantum states. Such scenarios occur when ignoring (ancilla) qubits in a quantum circuit, or when working with experimental setups. 

### Single Qubit

Let $p_j$ denote the discrete probability of having a single qubit state $\ket{\psi_j} = \alpha_j \ket{0} + \beta_j \ket{1}$. Then the _density_ operator is defined as

$$
    \rho = \sum_j p_j \ket{\psi_j}\bra{\psi_j}.
$$

As an example, suppose we have a devices that is used to initialize the state of a qubit to be $\ket{\psi} = \ket{0}$. However, the device is faulty: 99\% of the time it will generate the desired state $\ket{0}$, but 1\% of the time is will produce the state $\ket{1}$. Consequentially, if we were to repeatedly measure the state of our qubit right after initialization, using our previous pure-state ideas, we may mistakenly make the conclusion that our qubit is in a superposition $\ket{\psi} = \alpha \ket{0} + \beta \ket{1}$ where $|\alpha|^2 = 0.99$ and $|\beta|^2 = 0.01$ as shown in @fig:1-qubit-faulty-init. This of course is not the case, since with each experiment we actually have a pure state: $\ket{\psi} = \ket{0}$ or $\ket{\psi} = \ket{1}$. To capture this inconsistency, we instead write the state of our qubit as a \textit{density} or \textit{mixed} state:

$$
  \rho = \sum_j p_j \ket{\psi_j} \bra{\psi_j} \Rightarrow 0.99 \ket{0}\bra{0} + 0.01 \ket{1}\bra{1} = \begin{bmatrix}0.99 & 0 \\ 0 & 0.01 \end{bmatrix}
$$

The subsequent operation on the state are then described as

$$
\rho^\prime = U \rho U^\dagger
$$

where $U$ is some unitary gate.

The _purity_ of the state $\rho$ is defined as $\textrm{Tr}(\rho^2)$, and a state is pure if $\textrm{Tr}(\rho^2) = 1$. Moreover, since there is an associated classical uncertainty, _entropy_ can be defined as

$$
  S(\rho) = -\textrm{Tr}(\rho \log[\rho]) = -\sum_{k} \lambda_k \log \lambda_k
$$

where the $\log$ is base-2 and $\lambda_k$ are eigenvalues of $\rho$. The entropy tells us how much information we are missing (in bits) about the state; so a pure state has no entropy ($S(\rho) = 0$) since there is no missing information.

### Several Qubits

Continuing the example of an imperfect state initialization, suppose we
now try to initialize two qubits. The density matrix can then be
expressed as the four possible combinations, or as a tensor product of
each individual qubit density state.

$$
\rho & = (0.99 \cdot 0.99) \ket{00}\bra{00} \\
    & + (0.99 \cdot 0.01) \ket{01}\bra{01} \\
    & + (0.01 \cdot 0.99) \ket{10}\bra{10} \\
    & + (0.01 \cdot 0.01) \ket{11}\bra{11} \\
    & = \rho_A \otimes \rho_B
$$

where $\rho_A, \rho_B$ are the density states of each
individual qubit as defined previously.


To recover the density matrices $\rho_A$ or $\rho_B$ from the
composite state $\rho$, we take the \emph{partial trace}, e.g.:
$\rho_A = \textrm{Tr}_B \rho = \sum_i \bra{i}_B \rho \ket{i}_B$. Since the
state $\rho$ is formed by a tensor product, the subsequent partial
traces to recover $\rho_A$ and $\rho_B$ will simply reverse the
operation producing \ref{eq:mixed-qubit}. In general, $\rho$ might
encompass an entangled system, and in this case the partial trace is
guaranteed to be a mixed state.
