install_deps()
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


# Install relevant dependencies
install_deps = function() {
    # Install dependencies if needed
    message("Installing required R dependencies...")
    if (!require(ggplot2)) {
        install.packages("ggplot2")
    }
    if (!require(MASS)) {
        install.packages("MASS")
    }
    if (!require(ggplot2)) {
        install.packages("ggplot2")
    }
    message("Checking TensorFlow is installed properly")
    if (!require(tensorflow)) {
        install.packages("tensorflow")
        tensorflow::install_tensorflow()
    }
    tryCatch({
        tensorflow$tf$constant(1)
    }, error = function (e) {
        tensorflow::install_tensorflow()
    })
    message("Checking sgmcmc package installed")
    if (!require(sgmcmc)) {
        install.packages("sgmcmc")
    }
    # Get TensorFlow warnings out the way so output is more coherent
    quickSess = tf$Session()
    message("\n")
    # Create relevant directories if they do not exist
    for (f in c("gaussMix", "logReg", "nn", "plots")) {
        dir.create(f, showWarnings = FALSE)
    }
}
