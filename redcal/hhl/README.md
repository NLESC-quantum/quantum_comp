# Using HHL for Redundancy Calibration

## Contents

- [An exploratory notebook](./hhl_exploratory.ipynb), showing what we tried and how it worked.
- [A minimal version](./hhl_minimal.ipynb) of the document above, going directly to our best solution. This document is less transparent, and it was written for sending a simple calculation to qIBM servers.

## Conclusions

### TL;DR

After our preliminary analysis, we found the HHL algorithm is not the most promising choice for our needs.

---
### Details

The HHL algorithm presents some relevant limitations for us. Such as:

- Its results are approximate for matrices larger than $2x2$.
- It cannot discriminate the positive from the negative elements in the solution vector.

But the most important, by far, is the following: 

- It is quite hard to output the solution vector. The family of observables that the method accepts is limited. The most general one is `MatrixFunctional`, but don't get tricked by its promising name: it is not general, but a two-parameters tridiagonal matrix.

$$
{\tt MatrixFunctional} (\vec x; a, b) = \bra{x} B(a, b) \ket{x} = \vec x^T B(a, b) \vec x 
$$

where:

![](./img/tridiagonal.png)

For even more details, take a look at [this notebook](./hhl_exploratory.ipynb).