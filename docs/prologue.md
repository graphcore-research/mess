# Prologue

Regardless of the domain every new project appears to face a choice:

- build on top of a pre-existing framework
- start fresh and build a new framework

We think this is actually a false dichotomy but is still useful to demonstrate the
motivation for the MESS project. The curious reader may be wondering: There is no
shortage of software packages covering a broad range of electronic structure
simulations. Arguably there are too many packages that create a fractured ecosystem of
solutions that are not easily interopable with one another. This is not a new problem
and the work of MESS is likely making this problem worse not better!

The fundamental objective of MESS is to reimagine what a hybrid electronic structure and
machine learning framework might look like on future hardware architectures. This
viewpoint is what we think puts the **modern** in MESS. Beyond this our hope is to begin
to demystify the inner workings of electronic structure methods to help accelerate the
work of the molecular machine learning community.

To even begin to climb towards these lofty goals we start with a few constraints that we
think might help accelerate the ascent. These constraints are shamelessly borrowed from
what we think are some of the factors that have helped accelerate recent progress in
machine learning across multiple domains.

- hardware acceleration
- automatic differentiation
- high-level interpreted programming languages

On the last point, the reader who cut their teeth on electronic structure packages
implemented in Fortran programs that can trace their origins to the 1980s (or even
earlier!) may have the intuition that using an interpreted language would come with an
unacceptable performance penalty. This is a proven approach first introduced by MATLAB
{cite}`moler2020history` to accelerate numerical linear algebra before spreading
throughout computational sciences. NumPy {cite}`harris2020array` developed as an
open-source alternative to this array-centred programming framework and is now
completely ubiqutious throughout scientific computing.

```{bibliography}
:style: unsrt
```
