---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quantum Measurement Theory

One of the many things that make quantum systems interesting is the effects and consequences of measurement. 

Traditionally the description of measurement in quantum mechanics is terms of project measurements, namely if you have some observable $\hat{\Lambda}$ then it can be diagonalized using eigenvalues $\lambda$ and projection operators $\hat{\Pi_\lambda}$: $\hat{\Lambda} = \sum_\lambda \lambda \hat{\Pi_\lambda}$

The consequences are that for a pure state $\ket{\psi}$, the probability to obtain some $\lambda$ is $p_\lambda = \bra{\psi} \Pi_\lambda \ket{\psi}$. Likewise the conditional state would be $\ket{\psi_\lambda} = \Pi_\lambda \ket{\psi} / \sqrt{p_\lambda}$. For matrix states, the probability is given by $p_\lambda = Tr[\rho \hat{\Pi_\lambda}]$ and the conoditional state will then be $\rho_\lambda = \frac{\hat{\Pi_\lambda} \rho \hat{P_\lambda}}{p_\lambda}$

But what about the uncoditional state? I.e. what if one *makes the measurement, but ignores the result*. This will result in a state matrix: 

$$
\rho(T) = \sum_\lambda p_\lambda \rho_\lambda = \sum_\lambda \hat{\Pi_\lambda} \rho \hat{\Pi_\lambda}
$$

Projective measurement is interesting property because in general, it is an entropy-increasing process. Which is a sharp contrast to what happens classically: the unconditional classical state after non-disturbing measurement is identical to the state before the measurement. 

## Systems and meters

Unfortunately, projective measurement is inadequate. For one, there will be classical noise due to the measuring apparatus. Another more interesting reason is that there are measurement in which the conditional states will not be left in the eigenstate of the measured quantity. Example would be in photon counting, where measurement would leave a vacuum state. The primary reason for these problems is because measurement is usually not done on the system itself but rather observes what effect the system has on the environment.

So now we look at the combined system of our apparatus $\ket{\theta}$ and our system $\ket{\psi}$:

$$
\ket{\Psi} = \ket{\theta}\ket{\psi}
$$

Some entanglement procedure is done to couple the state:

$$
\ket{\Psi(T)} = \hat{U} \ket{\theta}\ket{\psi}
$$

Now, with a projective measurement $\hat{\Pi_r} = \ket{r}\bra{r} \otimes \hat{I}$, we would have the conditioned state:

$$
\Psi_r(T) = \frac{\ket{r}\bra{r} \hat{U} \ket{\theta}\ket{\psi}}{\sqrt{p_r}}
$$

The measurement disentangles the system and the apparatus, so we can write it as:

$$
\Psi_r(T) = \ket{r} \frac{\hat{M_r} \ket{\psi}}{\sqrt{p_r}}
$$

where $\hat{M_r} = \bra{r}\hat{U}\ket{\theta}$ and is called the measurement operator. The probability distribution for would then be $p_r = \bra{\psi}\hat{M_r}^\dagger \hat{M_r} \ket{\psi}$

Now, because the system and apparatus are no longer entangled, everything can be viewed purely with the measurement operators. E.g.: $\psi(T) = \frac{\hat{M_r} \ket{\psi}}{\sqrt{p_r}}.$

If we just make one measurement, then the conditioned state is $\ket{\psi_r}$ would be not of great interest. If did a sequence of measurements, then the state matrix of the conditioned state would be:

$$
\rho_r =  \frac{\hat{M_r} \rho \hat{M_r}^\dagger}{p_r} = \frac{\mathcal{J}[\hat{M_r}] \rho}{p_r}
$$

What if we ignored the results? Well then the state would be:

$$
\rho = \sum_r p_r \rho(T) = \sum_r \mathcal{J}[\hat{M_r}] \rho_r.
$$

What this all tells us, is that preforming non-projective measurement, we have no guarantee that repeating the measurement will yield the same result. In fact, the final state may be completely unrelated to the initial state or the results obtained.

## Example: Quantum Teleportation

Alice (A) wants to send Bob (B) a qubit. Alice owns a qubit that she wants to send as well as a qubit that is entangled with Bob's qubit.

Create an entangled pair:

$$
\frac{1}{\sqrt{2}} \left( \left | 00 \right \rangle + \left | 11 \right \rangle \right)
$$

```{code-cell} ipython3
:tags: [remove-cell]
import numpy as np
```

```{code-cell} ipython3
EPR = np.array([[1], [0], [0], [1]])/np.sqrt(2)
print(EPR)
```

Create an arbitrary state $(\alpha \left | 0 \right \rangle + \beta \left | 1 \right \rangle)$  that Alice will send to Bob. In this case the state is simply:

$$
\left | \phi \right \rangle = \frac{1}{\sqrt{2}} \left( \left | 0 \right \rangle - \left | 1 \right \rangle \right)
$$

```{code-cell} ipython3
phi = np.array([[1], [-1]])
phi = phi /np.linalg.norm(phi)
print(phi)
```

Now the state of the three qubits looks like:

$$
\left | \psi \right \rangle =   \left | \phi \right \rangle \otimes \frac{1}{\sqrt{2}} \left( \left | 00 \right \rangle + \left | 11 \right \rangle \right)
$$

```{code-cell} ipython3
# Combine the psi and EPR states to the full space
psi = np.kron(phi, EPR)
print(psi)
```

Now let us preform the quantum teleportation. The first stage initializes the qubits; one which denotes what we want to send, and the other initialization for constructing an EPR pair. The second stage entangles Alice's qubit with the qubit from the EPR pair which Alice owns -- this is the main "teleportation" step. 

```{code-cell} ipython3
# Manually construct matrices and preform evolution
# Define CNOT operation
CNOT = np.eye(4)
CNOT[2,2] = 0; CNOT[2,3] =1; CNOT[3,2] = 1; CNOT[3,3] = 0
CNOT = np.kron(CNOT, np.eye(2))

# Define hadamard operation
HADAMARD = np.array([[1,1], [1,-1]])/np.sqrt(2)
IDENTITY = np.kron(np.eye(2), np.eye(2))
HADAMARD = np.kron(HADAMARD, IDENTITY)
```

After performing all the operations we have a complete state (which is pure):

```{code-cell} ipython3
psi = CNOT.dot(psi)
psi = HADAMARD.dot(psi)
print(psi)
```

### Measurement

In the quantum teleportation protocol, we would then measure both of Alice's qubits and then send to Bob so that he can recover the sent state. But why do we need the classical bits? Generally this is motivated by showing we will have 4 possibilities after measuring Alice's qubits.

Let's view through the lens of __quantum measurement__. What if we *measure but ignore the results* of Alice's qubits. Well now our pure states becomes a state matrix.

In particular, we have an observable operator ($\hat{\Lambda}$) which can be diagonalized as:

$$
\hat{\Lambda} = \sum_\lambda \lambda \Pi_\lambda
$$

The $\Pi_\lambda$ are *projective measurements* and $\lambda$ are the eigenvalues.

So by performing a measurement and ignoring the results, we take our pure state $\rho_a = \left | \psi \right\rangle \left \langle \psi \right |$ and created a mixed state: $\rho_b = \sum_\lambda \Pi_\lambda \rho_a \Pi_\lambda$

Moreover, this shows that, in general, a measurement is an entropy-increasing process, unless we *keep track* of the measurement results.

#### Classical Measurement
This result is different from what happens classically. When we have a classical state and perform a measurement, we can model it using Bayesian inference (which captures the *conditional*, disturbing, nature of the system - where the *state* of a system changes *given* the measurement outcome.) Using this model, non-disturbing measurements show that a-posteriori state is identical to the a-priori state. 

```{code-cell} ipython3
# Preform the projective measurement
rho = np.zeros((8,8))
for i in range(4):
    P_i = np.zeros((8,8)); P_i[i*2,i*2] = 1; P_i[i*2+1,i*2+1] = 1
    p_i = psi.conj().T.dot(P_i.dot(psi))
    after_measure = P_i.dot(psi) / np.sqrt(p_i)
    rho = rho + p_i * after_measure.dot(after_measure.conj().T)
rho
```

Let us take a look at the purity of the system now as compared to before measurement. We can see entropy was increased!

```{code-cell} ipython3
r = np.outer(psi.conj().T, psi)
print("Before measurement")
print(np.trace(r.dot(r)))
print("After measurement")
print(np.trace(rho.dot(rho)))
```

Moreover, let use trace out Alice's system and only look at Bob's system. This will give us insight to what Bob currently knows (after the *measurement but ignoring the results*.)

```{code-cell} ipython3
tr_A = np.zeros((2,2))

# By linearity in trace input
for i in range(4):
    # |b_1><b_2| <00|00>{<01|01|}{10|10}{11|11} = is already encoded in rho just find |b_1><b_2| (which are just blocks in rho)
    tr_i = rho[np.ix_([i*2,i*2+1], [i*2, i*2+1])]
    tr_A = tr_A + tr_i
```

```{code-cell} ipython3
print('Bob\'s state matrix:')
print(tr_A)

print('Clearly, this is maximally mixed:')
# Mixed state since < 1
print(np.trace(tr_A.dot(tr_A)))
```

As we can see, Bob's state is $\frac{1}{2} \mathbf{I}$.

Well this is a maximally-mixed state! Namely, Bob's state is mixture of states that all occur with equal probability; we have no clue what those states are, as there are infinitely many ways to construct (summing to) the identity.

Now going back, if we did record the results then we would have 4 possible combinations (tracked by classical bits) and likewise we would know exactly the 4 pure states that Bob has, each occurring with probability $\frac{1}{4}$. And hence we know what the summation to produce the identity!

### Conclusion
This mathematically shows the interesting ideas concerning *quantum measurement*. And more importantly, the difference between classical and quantum measurement -- where classically, if we ignore the measured result (i.e. non-disturbing), our a-posteriori *state* (how much we *know* about a system) does not change, where as in quantum it does! 