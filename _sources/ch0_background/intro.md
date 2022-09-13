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

# About this tutorial

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is present in all the notebooks.
# It makes the necessary packages available and adjusts various settings.
# You should execute this cell at the start.


#init_notebook()

#from IPython.display import HTML
```

In this tutorial we seek to provide an introduction to the topic of noise in quantum computers, and particularly, strategies to _characterize and mitigate_ this noise.
We want this tutorial to be accessible and useful to people with different backgrounds and motivations.

We specifically hope this tutorial is useful to 
- someone working directly in increasing the performance of quantum computers
- someone wanting to perform calculations on contemporary (noisy) quantum computers
- someone who simply wants to learn


Finally, we also want this tutorial to be equally useful if you are, say, a professor and you want to apply the ideas and just need a quick overview of what research activity is there.

We hope that you find something useful, and we always appreciate your questions and feedback. You can open an issue, (or even make a pull request right away).

## Structure

The tutorial is meant to be _cohesive_ -- at the bare minimum links are provided to resources to explore the topics outside the scope of this tutorial.

Topics covered:
1. Introduction to Quantum Computing
2. Noise Characterization
3. Noise Mitigation
4. Software and Quantum Toolchains

## Prerequisites

### Background knowledge

While the math that we use only requires linear algebra and calculus, this tutorial is complex. We will work directly with the Schrödinger equation, and various quantum theories. We provide an introduction to hopefully ease the learning curve to the math, but those without any previous exposure to these mathematical and quantum contexts will find the tutorial difficult to truly appreciate.

### Code

We provide source code for all the computer simulations used in the tutorial as well as suggestions of what you can investigate on your own. In order to use these, you need to be familiar with Python. Additionally we make use of a number of python packages:

- numpy
- scipy
- qutip
- qiskit
- mitiq


+++