# doubly_stochastic_lif_inference

Library for fitting a population of LIF neurons driven by a one dimensional
common input. The input consists of white noise and a mean which is stochastic
itself (hence doubly stochastic).

## Usage

A minimal example is given in the file example.py, including data simulation
from the generative model, fitting the model, and plotting the results. The main
function, that should interest the general user, is contained in model_fit.py.
The code was written with Python 2.7.

## Remarks

While the code is working well for simulated and invitro data, the model has
problems with spike sorting errors in invivo recordingds (in particular with
unreasonably short interspike intervals < 3ms). If problems are encountered,
removing the spikes at the end of these intervals could help, and also
adjusting the parameter sorting_error to a plausible value.

## Required libraries

Required dependencies are: numpy, numba, scipy, multiprocessing, functools,
warnings.

## Authorship and Contact

The code was developed by Christian Donner and Josef Ladenbauer.
For questions contact christian.research(at)mailbox.org.

## License

Copyright (C) 2019, Christian Donner

This file is part of doubly_stochastic_lif_inference.

doubly_stochastic_lif_inference is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

doubly_stochastic_lif_inference is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
License for more details.

You should have received a copy of the GNU General Public License
along with doubly_stochastic_lif_inference.  If not, see
<http://www.gnu.org/licenses/>.