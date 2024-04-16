# MESS: Modern Electronic Structure Simulations

:::{note}
This project is a work in progress.
Software features marked with a 🌈 indicate that
this functionality is still in the planning phase.
:::

Welcome to MESS, a python framework for exploring the exciting interface
between machine learning, electronic structure, and algorithms.
Our main focus is building a fully hackable implementation of
[Density Functional Theory (DFT)](https://en.wikipedia.org/wiki/Density_functional_theory).

Target applications include:
* high-throughput DFT simulations for efficient large-scale molecular dataset generation
* exploration of hybrid machine learned/electronic structure simulations

Within DFT there are many different approximations for handling quantum-mechanical
interactions.  These are collectively known as exchange-correlation functionals and
MESS provides a few common implementations:
* LDA, PBE, PBE0, B3LYP
* dispersion corrections 🌈
* machine-learned exchange-correlation functionals 🌈

This project is built on
[JAX](https://jax.readthedocs.io/en/latest/) to support rapid
prototyping of high-performance simulations.  MESS benefits from many features of JAX:
* Hardware Acceleration
* Automatic Differentiation
* Program transformations such as JIT compilation and automatic vectorisation.
* Flexible floating point numeric formats


## Minimal Example

Calculate the ground state energy of a single water molecule using the 6-31g basis set
and the [local density approximation (LDA)](https://en.wikipedia.org/wiki/Local-density_approximation):
```python
from mess import Hamiltonian, basisset, minimise, molecule

mol = molecule("water")
basis = basisset(mol, basis_name="6-31g")
H = Hamiltonian(basis, xc_method="lda")
E, C, sol = minimise(H)
E
```

## Next Steps

::::{grid}
:gutter: 2

:::{grid-item-card} {material-regular}`map;2em` Learn
:link: tour
:link-type: doc
:::

:::{grid-item-card} {material-regular}`construction;2em` Build
:link: api
:link-type: doc

:::

::::
