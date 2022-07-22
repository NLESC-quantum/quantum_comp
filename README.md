# quantum_comp

Knowledge collection of the quantum computing exploration

## Learning materials

### Books and tutorials

- [QisKit textbook](https://qiskit.org/textbook/preface.html)
- [Lecture notes by Ronald de Wolf](https://homepages.cwi.nl/~rdewolf/qcnotes.pdf)
- [Quantum computing since Democritus](https://www.scottaaronson.com/democritus/)
- [Quantum Computation and Quantum Information. Isaac Chuang](https://www.bol.com/nl/nl/p/quantum-computation-and-quantum-information/1001004010977341/?s2a=)
- [Modern quantum mechanics. J.J. Sakurai](https://www.bol.com/nl/nl/p/modern-quantum-mechanics/9300000013146000/). Not to be mistaken with Advanced quantum mechanics, by the same author.

### Mathematics

- Geometrical approache to the [Moore-Penrose generalized inverse](https://www.cantorsparadise.com/demystifying-the-moore-penrose-generalized-inverse-a1b989a1dd49).

### Publications

- Commodity compute and data-transport system design in modern large-scale distributed radio telescopes. [Chris Broekema's PhD thesis](https://www.astron.nl/~broekema/Thesis/PhD-Thesis.pdf)
- [Nice review of some applications of QIP](https://arxiv.org/pdf/2203.01831.pdf)

### Tools
- [QisKit](https://qiskit.org/)
- [Pennylane](https://pennylane.ai/)
- [NANDgame](https://nandgame.com/). Great for learning about **classical** logic gates.

### Tips
In order to use bra-ket notation in Jupyter notebooks, include the snippet below in a markdown cell:

```markdown
$$
\newcommand{\braket}[2]{\left\langle{#1}\middle|{#2}\right\rangle}
\newcommand{\ket}[1]{\left|{#1}\right\rangle}
\newcommand{\bra}[1]{\left\langle{#1}\right|}
$$
```

it will render to whitespace, but it will allow to use the latex commands `\bra{q}`, `\ket{q}` and `\braket{q, p}`.
