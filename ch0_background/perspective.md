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

This chapter presents a (biased) perspective of quantum computing, with emphasis on contemporary quantum computers.

## Why Quantum Computers?

Often termed as the _first quantum revolution_, the technological contributions that lead to computers, optical fiber communication, and GPS were based on the inventions in the 20th-century, including: transistors, lasers, and atomic clocks. Although powerful, these technologies do not harness the full power of quantum mechanics.

Now let us consider the rich field of [Nuclear magnetic resonance (NMR)](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance). This is a perhaps an excellent example of harnessing the power of quantum mechanics, whereby a nuclei with _spin_ (a quantum object) interacts with an external magnetic field. We can control the spin, a quantum object, by tailoring the magnetic field. In principle, we could use this as our _quantum computer_: we start in an initial spin state, apply a magnetic field, and get a final spin state that encodes a solution to our problem of interest.

```{code-cell} ipython3
:tags: ["remove-input"]

example = [{
        "question": "Would such a setup be a good quantum computer?",
        "type": "multiple_choice",
        "answers": [
            {
                "answer": "Yes",
                "correct": False,
                "feedback": "Incorrect"
            },
            {
                "answer": "No",
                "correct": True,
                "feedback": "Correct"
            }
        ]
    }]

display_quiz(example)
```

Ignoring the many engineering challenges such as making many spins talk with one another, there is a serious scalability issue. If we wanted to reach any arbitrary spin state, then we would need to figure out the magnetic field parameters that would achieve that spin state. This would require *simulating the quantum system* -- which although is feasible for small systems, it becomes exponentially difficult to do so with increasing system dimension.

```{note}
Suppose we have 100 qubits (2-dimensional quantum system) then the simulation boils down to computing with a ($2^{100} \times 2^{100}$) matrix. So finding the parameters of a tunnable magnetic field is infeasible.
```

But what if we precomputed a small set of magnetic field parameters that achieved a specific quantum evolution, and then applied them in a specific order to reach an arbitrary spin state? This idea is exactly given by the [Solovay-Kitaev theorem](https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem), and is perhaps _the most fundamental result in quantum computing_. The theorem essentially tells us that if our small set of magnetic field parameters (quantum gates) satisfy _universality conditions_ then any arbitrary operation can be approximated with a finite number $(O(m\log ^{c}(m/\varepsilon )))$ of gates (a quantum circuit).

> TL;DR: Thanks to the Solovay-Kitaev theorem, a quantum computer allows us to scalably perform any arbitrary quantum operation with a pre-calibrated number of _universal_ quantum operations (aka gates).