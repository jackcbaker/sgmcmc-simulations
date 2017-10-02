# Simulations for [`sgmcmc` package](https://github.com/STOR-i/sgmcmc)

Simulations for the package [`sgmcmc`](https://github.com/STOR-i/sgmcmc) available for `R`. The simulations are for the companion paper available from [arXiv](https://arxiv.org) (see Section 5).


### Running the simulations

To run the simulations simply run the script `runSimulations.R`. The plot output corresponding to each simulation will appear in the directory `plots`.


### Dependencies

The simulations depend on the following `R` packages: `sgmcmc` (the package itself), `rstan`, `MASS`, `ggplot2` (all available on `CRAN`). These dependencies should be automatically installed when `runSimulations.R` is run. The script will also check `TensorFlow` for `R` (also available on `CRAN`) has been installed properly, which is a dependency for `sgmcmc`. If it is not, it will attempt to install this itself. The installation of `TensorFlow` for `R` requires `python-pip` and `python-virtualenv` to be installed so if these are not available the script will stop and prompt you to install these.


### System requirements

Some of the simulations load large datasets into memory, so it's recommended you have at least 8GB of RAM.


### Run time

Simulations took approximately 3 hours on a quad core i5 laptop with 8GB RAM.


### Troubleshooting

`Error: Python module tensorflow was not found.` -- Try rerunning `runSimulations.R` or restarting your `R` session. This Error can sometimes occur when `TensorFlow` is installed during the `runSimulations.R` script and then is immediately called. Restarting the `R` session and rerunning the script should solve the error.


### Original Setup Details

Original results were run on a laptop with Ubuntu 16.04 LTS; `R` version 3.2.3; `Python` version 2.7.12; `TensorFlow` version 1.3.0; `TensorFlow` for `R` version 1.4.0; `sgmcmc` version 0.2.0. 

Results were reproduced on a linux cluster running Ubuntu 14.04 LTS; `R`version 3.2.3; `Python` version 2.7.6; `TensorFlow` version 1.3.0; `TensorFlow` for `R` version 1.4.0; `sgmcmc` version 0.2.0.


### Remarks

While we can guarantee reproducibility on a single platform, and have ensured reproducibility across two platforms (see Original Setup Section). The `TensorFlow` seed setting appears to be very dependent on version and platform, this can make reproducibility across different platforms difficult. We have done our best to make everything as reproducible as possible and have given as much detail as possible of our set up in the Original Setup Section to make it easier for the reviewer.
