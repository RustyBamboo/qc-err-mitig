# Measurement Error

So far the discussion has been focused purely on gathering information
of a quantum system. However, a large source of error in current devices
(1-5\% error rate) are from measurement errors. Naive methods in quantum state (process) tomography are susceptible to
such measurement errors, and more generally to state preparation and
measurement (SPAM) errors.

A straightforward but effective classical model for measurement error is
a Stochastic Assignment matrix

$$
A_{i,j} = P(i|j)
$$

which maps the probability of obtaining an incorrect output $i$
given a correct input $j$, where $i$ and $j$ are bitstrings
associated with a qubit measurement.

A complete characterization of the $A$-matrix requires preparing
$2^n$ states. For example, to characterize two qubits, four states are
needed: $\{\ket{00}, \ket{01}, \ket{10}, \ket{11}\}$. If the errors
are uncorrelated, then the $A$-matrix can be constructed as a tensor
product. In this case, only 2 states are needed as all probabilities can
be estimated simultaneously by preparing all-zero and all-one state.
Although the tensor product model is simple, cross-talk errors that are
present in real-world setups are ignored. A recent work has introduced a
model that takes into account a limited measure of correlated noise
based on Continuous Time Markov Processes (CTMP) [arXiv:2006.14044](https://arxiv.org/abs/2006.14044).
The matrix then has a form $A = e^G$ where $G$ is a sum of local
operators the generate one and two-qubit readout errors. The total cost
of this approach is $O(e^{5\gamma} poly(n))$ where $\gamma$ is
related to noise strength and assumed small. The authors compare the
tensor product noise model and the CTMP noise model with a full
characterization model for 4,5,6,7 qubits.