# Install relevant dependencies
installDeps = function() {
    suppressPackageStartupMessages({
        # Install dependencies if needed
        message("Installing required R dependencies...\n")
        if (!require(ggplot2, quietly = TRUE)) {
            install.packages("ggplot2", repos = "https://cloud.r-project.org")
        }
        if (!require(MASS, quietly = TRUE)) {
            install.packages("MASS", repos = "https://cloud.r-project.org")
        }
        message("\nChecking TensorFlow is installed properly...\n")
        if (!require(tensorflow, quietly = TRUE)) {
            install.packages("tensorflow", repos = "https://cloud.r-project.org")
            tensorflow::install_tensorflow()
        }
        tryCatch({
            tensorflow::tf$constant(1)
        }, error = function (e) {
            tensorflow::install_tensorflow()
        })
        message("\nChecking sgmcmc package installed\n")
        if (!require(sgmcmc, quietly = TRUE)) {
            install.packages("sgmcmc", repos = "https://cloud.r-project.org")
        }
        # Get TensorFlow warnings out the way so output is more coherent
        quickSess = tensorflow::tf$Session()
        message("\n")
        # Create relevant directories if they do not exist
        for (f in c("gaussMix", "logReg", "nn", "plots")) {
            dir.create(f, showWarnings = FALSE)
        }
    })
}

# Run simulations
installDeps()
source("plots.R")
message("\n##########\nRunning usage examples (Section 4)\n")
source("usage.R")
runSimulations()
message("\n##########\nRunning simulations for Gaussian Mixture (Section 5.1)\n")
source("gaussMix.R")
runSimulations()
plotGM()
message("\n##########\nRunning simulations for Logistic Regression (Section 5.2)\n")
source("logReg.R")
runSimulations()
plotLogReg()
message("\n##########\nRunning simulations for Bayesian Neural Network (Section 5.3)\n")
source("nn.R")
runSimulations()
plotNN()
