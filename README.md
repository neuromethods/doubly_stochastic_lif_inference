# inference_for_doubly_stochastic_IF_models

Library for fitting a population of doubly-stochastic integrate-and-fire neurons 
to spike train data. The model accounts for fast independent and slower shared input
fluctuations that dominate the low-dimensional collective dynamics. In particular, 
each neuron is driven by an independent Gaussian white noise process, whose mean varies 
according to a slower stochastic process that is shared among the population. 
The statistical inference method is described in: __Donner, Opper, Ladenbauer,__ 
___Inferring the collective dynamics of neuronal populations from single-trial spike trains 
using mechanistic models___ (under review)

## Usage

An example is given in the file example.py, which includes generation of synthetic data
from the generative model, parameter estimation, and visualization of the results. 
The code was written with Python 2.7.

## Remarks

Unreasonably small neuronal interspike intervals (ISIs < 3ms), due to spike sorting errors 
from in-vivo recordings for example, may cause problems. In this case we recommend to remove 
very small ISIs and/or set the parameter sorting_error to a larger value.

## Required libraries

Required dependencies are: numpy, numba, scipy, multiprocessing, functools,
warnings.

## Authorship and Contact

The code was developed by Christian Donner and Josef Ladenbauer.
For technical questions please contact christian.research(at)mailbox.org.
