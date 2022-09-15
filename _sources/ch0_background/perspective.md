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

```{code-cell} ipython3
:tags: [remove-cell]
from jupyterquiz import display_quiz
```



# Setting the Perspective

This chapter presents a _biased_ perspective of quantum computing, with emphasis on contemporary quantum computers.

## Why Quantum Computers?

Often termed as the **first quantum revolution**, the technological contributions that lead to computers, optical fiber communication, and GPS were based on the inventions in the 20th-century, including: transistors, lasers, and atomic clocks. Although powerful, these technologies do not harness the full power of quantum mechanics. 

```{admonition} Example
:class: tip
[Diodes](https://en.wikipedia.org/wiki/Tunnel_diode) make use of the [quantum tunneling](https://en.wikipedia.org/wiki/Quantum_tunnelling) effect.
```

Now let us consider the rich field of [Nuclear magnetic resonance (NMR)](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance). This is a perhaps an excellent example of harnessing the power of quantum mechanics, whereby a nuclei with _spin_ (a quantum object) interacts with an external magnetic field. We can control the spin, a quantum object, by tailoring the magnetic field. In principle, we could use this as our _quantum computer_: we start in an initial spin state, apply a magnetic field, and get a final spin state that encodes a solution to our problem of interest.

```{code-cell} ipython3
:tags: ["remove-input"]

example = [{
        "question": "Would such a setup be a good quantum computer?",
        "type": "multiple_choice",
        "answers": [
            {
                "answer": "No",
                "correct": True,
                "feedback": "Correct"
            },
            {
                "answer": "Yes",
                "correct": False,
                "feedback": "Incorrect"
            }
        ]
    }]

display_quiz(example)
```

Ignoring the many engineering challenges (such as making many spins talk with one another), there is a serious scalability issue. If we wanted to reach any arbitrary spin state, then we would need to figure out the magnetic field parameters that would achieve that spin state. This would require *simulating the quantum system* -- which although is feasible for small systems, it becomes exponentially difficult to do so with increasing system dimension.

```{note}
Suppose we have 100 qubits (2-dimensional quantum system) then the simulation boils down to computing with a ($2^{100} \times 2^{100}$) matrix. So finding the parameters of a tunnable magnetic field is infeasible.
```

But what if we precomputed a small set of magnetic field parameters that achieved known quantum evolutions, and then applied them in a specific order to reach an arbitrary spin state? This idea is exactly given by the [Solovay-Kitaev theorem](https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem), and is perhaps _the most fundamental result in quantum computing_. The theorem essentially tells us that if our small set of magnetic field parameters (quantum gates) satisfy _universality conditions_ then any arbitrary operation can be approximated with a finite number $(O(m\log ^{c}(m/\varepsilon )))$ of gates (a quantum circuit).

> TL;DR: Thanks to the Solovay-Kitaev theorem, a quantum computer allows us to scalably perform any arbitrary quantum operation with a pre-calibrated number of _universal_ quantum operations (aka gates).

## Noise and Noise Mitigation

As we enter the **second quantum revolution**, our goal is to _accurately_ control quantum systems.  Unfortunately, unwanted interactions will disturb the overall system. So even if we are able to calibrate a universal gate set and employ the Solovay-Kitaev theorem, we may be unable to reach our desired state. 

Of course, to overcome these concerns, the goal is always to engineer better hardware. But in reality building better quantum hardware is rather hard and expensive to do. Ok, well then maybe we can encoding quantum information in a special subspace and employ [quantum error correction](https://en.wikipedia.org/wiki/Quantum_error_correction). Although powerful, there is an incredible overhead in the number of qubits needed to perform the encoding. Moreover, [thresholds](https://en.wikipedia.org/wiki/Quantum_threshold_theorem) on the physical error rates are relatively uncertain.

```{note}
Noise was a major early criticism of quantum computing, however quantum error correcting codes were quickly discovered. Essentially you can use multiple physical qubits together to act as one "logical" qubit with better coherence properties than any of the individual qubits. These error correcting codes scale very favourably, so the decoherence of the logical qubit will decrease exponentially with the number of physical qubits (assuming a low enough noise ratio). The exponential error suppression means that decoherence is no longer seen as a fatal problem for quantum computing: though there is a large scope towards satisfying the caveats for the error correction (which is a difficult engineering problem).
```

The situation seems rather bleak: we can't create _perfect_ quantum hardware, and quantum error correction is impractical. _Is there something we can do?_ 

Well, this leads us to the purpose of this tutorial -- to make use of _noisy_ quantum computers as much as we can. Let us consider where we could improve quantum performance, as shown in the following figure:

```{image} mitigation.drawio.svg
:align: center
```

1. Low level: optimize pulses, exploit hardware-specific quantum features
2. Middle level: design noise-resilient quantum algorithms, and aposterori noise-cancellation techniques
3. High level: when employing the Solovay-Kitaev theorem, consider noise reduction as an additional parameter (aka noise-aware compilers)

> TL;DR: In this tutorial, we will specifically focus on the _low level_ and _middle level_ strategies. We believe that contemporary quantum computers of today will receive the most benefit from optimization in these two layers. 