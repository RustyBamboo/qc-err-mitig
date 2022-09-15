# Randomness

[Measure theory](https://en.wikipedia.org/wiki/Measure_(mathematics)) studies things that are measurable (e.g. length, area, volume) but generalized to various dimensions. The _measure_ tells us how things are distributed and concentrated in a mathematical space.

::::{admonition} Example: The Sphere
Any point on the sphere can be specified by three parameters: e.g. (x,y,z) in Cartesian coordinates or $(\rho, \theta, \phi)$ in spherical coordinates.

Suppose we wish to find the volume. The first thought may be to integrate each parameter over its full range:

$$
V = \int_0^{r} \int_0^{2\pi} \int_0^{\pi} d\rho~ d\phi~ d\theta = 2\pi^2 r
$$

But this is clearly wrong (the volume of a sphere with radius $r$ is $\frac{4}{3}\pi r^3$.) Taking the integral in this way does not account for the _structure_ of the sphere with respect to the parameters. If you have ever studied multivariate calculus this may sound familiar.

```{image} volume-element.png
:align: center
```

The issue is that although the difference $d\theta$ and $d\phi$ are the same, there is more _stuff_ near the equator of the sphere. Therefore the integral must account for the value of $\theta$. Similarly, as we are closer to the center of the sphere there is less _stuff_ compared to as we go further away from the center. Hence we must also include the radius $\rho$ as part of the integral. 

So the correct integral is:

$$
V = \int_0^r \int_0^{2\pi} \int_0^{\pi} \rho^2 \sin \theta~ d\rho~ d\phi~
d\theta = \frac{4}{3}\pi r^3
$$

The extra terms $\rho^2\sin\theta$ is the _measure_. The measure weight portions of the sphere differently depending on where they are in the sphere.

::::

Although the _measure_ tells us how to properly integrate, is also tells us another important fact: _how to sample points in the spae uniformly as random_. We cannot simply take each parameter and sample it from the uniform distribution -- this does not take into account the spread of the space. The measure tells us the distribution of each parameters and let us obtain something truly uniform.

## Haar Measure

Quantum evolutions (and operations in quantum computing) are described by unitary matrices. Just like the sphere, unitary matrices can be expressed in terms of a fixed set of parameters. For example, a qubit operation can be written as:

$$
\begin{align}
U(\alpha, \theta, v_x, v_y, v_z) = e^{i\alpha} \exp(-i \theta/2 \vec{v} \cdot \vec{\sigma}) &= \exp[-i \theta/2 (v_x \sigma_x + v_y \sigma_y + v_z \sigma_z)] \\
&= \cos(\theta/2) I - i \sin(\theta/2) (v_x \sigma_x + v_y \sigma_y + v_z \sigma_z)
\end{align}
$$

where $v_x^2 + v_y^2 + v_z^2 = 1$. We ignore $e^{i\alpha}$ because it is a global phase. So we are left with four parameters. But because we have the constraint that $v_x^2 + v_y^2 + v_z^2 = 1$, we only need a total of **3 parameters** to describe a single qubit operation.

In fact we can write any unitary matrix (ignoring global phase) of size $N \times N$ as:

$$
U(\vec{l}) = \exp(-i \vec{l} \cdot \vec{\Lambda})
$$

where $\vec{\Lambda}$ contains $N^2 -1$ operators that define space, similar to the Pauli matrices for the case $N=2$. 


For a dimension $N$, the unitary matrix of size $N \times N$ form the unitary group $U(N)$. 

```{note}
Because we ignored the global phase, the operations we provided are technically $SU(N)$ the _special_ unitary group, where $det(U) = 1$. 
```

Just like the points of the sphere, we can perform operations on the unitary group: such as integration, functions, and sample uniformly. But just like the sphere, we have to add the _measure_ in order to properly account for the different weights in different regions of the space. This is known as the _Haar measure_, is often denoted as $\mu_N$.

$\mu_N$ tells us how to weight the elements of $U(N)$. So for example, if we have some function $f$ acting on the elements of $U(N)$ and we would like to integrate over this group, we would have:

$$
\int_{V \in U(N)} f(V) d\mu_N(V).
$$

## Example: qubit