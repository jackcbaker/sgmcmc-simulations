# Simulations for [`sgmcmc` package](https://github.com/STOR-i/sgmcmc)

Simulations for the package [`sgmcmc`](https://github.com/STOR-i/sgmcmc) available for `R`. The simulations are for the companion paper available from [arXiv](https://arxiv.org) (see Section 5).


### Running the simulations

To run the simulations simply run the script `runSimulations.R`. The plot output corresponding to each simulation will appear in the directory `plots`.


### Dependencies

The simulations depend on the following `R` packages: `sgmcmc` (the package itself), `rstan`, `MASS`, `ggplot2` (all available on `CRAN`). These dependencies should be automatically installed when `runSimulations.R` is run. The script will also check `TensorFlow` for `R` (also available on `CRAN`) has been installed properly, which is a dependency for `sgmcmc`. If it is not, it will attempt to install this itself. The installation of `TensorFlow` for `R` requires `python-pip` and `python-virtualenv` to be installed so if these are not available the script will stop and prompt you to install these.


### System requirements

Some of the simulations load large datasets into memory, so it's recommended you have at least 8GB of RAM.


### Run time

Simulations took approximately 3 hours on four 2.3GHz Intel Xeon cores.
