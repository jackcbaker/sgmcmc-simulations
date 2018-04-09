### Run simulations in Section 5.1 ###
library(sgmcmc)
library(MASS)
library(rstan)

# Declare log likelihood
logLik = function( params, dataset ) {
    # Declare Sigma (assumed known)
    SigmaDiag = c(1, 1)
    # Declare distribution of each component
    component1 = tf$distributions$MultivariateNormalDiag( params$theta1, SigmaDiag )
    component2 = tf$distributions$MultivariateNormalDiag( params$theta2, SigmaDiag )
    # Declare allocation probabilities of each component
    probs = tf$distributions$Categorical(c(0.5,0.5))
    # Declare full mixture distribution given components and allocation probabilities
    distn = tf$distributions$Mixture(probs, list(component1, component2))
    # Declare log likelihood
    logLik = tf$reduce_sum( distn$log_prob(dataset$X) )
    return( logLik )
}

# Declare log prior
logPrior = function( params ) {
    # Declare hyperparameters mu0 and Sigma0
    mu0 = c( 0, 0 )
    Sigma0Diag = c(10, 10)
    # Declare prior distribution
    priorDistn = tf$distributions$MultivariateNormalDiag( mu0, Sigma0Diag )
    # Declare log prior density and return
    logPrior = priorDistn$log_prob( params$theta1 ) + priorDistn$log_prob( params$theta2 )
    return( logPrior )
}

runSimulations = function() {
    # Set stepsizes for each method, run each method for seed 2
    stepsizeList = list("sgld" = 5e-3, "sghmc" = 5e-4, "sgnht" = 3e-4, "sgldcv" = 1e-2, "sghmccv" = 1.5e-3, "sgnhtcv" = 1e-3)
    methods = c("sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv")
    plotList = list()
    cat("sgmcmc function: ")
    for (method in methods) {
        cat(paste0(method, " "))
        stepsize = stepsizeList[[method]]
        runSeed(2, method, stepsize)
    }
    cat("\n")
    # Run RSTAN (Hamiltonian Monte Carlo run) as approximation to 'truth' for comparison
    hmc(2)
}

runSeed = function(testSeed, method, stepsize) {
    # Generate test data given seed
    set.seed(testSeed)
    testData = genTestData()
    # Set base arguments for sgmcmc functions
    argsCurr = list( "logLik" = logLik, "dataset" = testData$dataset, "params" = testData$params,
            "logPrior" = logPrior, "minibatchSize" = testData$minibatchSize, nIters = 2 * 10^4, 
            "verbose" = FALSE, "seed" = testSeed, "stepsize" = stepsize )
    # Set additional optimisation argument for sgldcv
    if (grepl("cv", method)) {
        argsCurr$optStepsize = testData$optStepsize
        # Reduce number of iterations for sghmccv due to 5x computational cost
        if (method == "sghmccv") {
            argsCurr$nIters = 2000
        } else {
            argsCurr$nIters = 10^4
        }
        # Call current sgmcmccv function, referred to as `method`, remove burn-in and store
        output = do.call(method, argsCurr)
        rmBurnIn(output, method, stepsize)
    } else {
        # Reduce number of iterations for sghmc due to 5x computational cost
        if (method == "sghmc") {
            argsCurr$nIters = 4000
        } else {
            argsCurr$nIters = 2 * 10^4
        }
        # Call current sgmcmc function, referred to as `method`, remove burn-in and store
        output = do.call(method, argsCurr)
        rmBurnIn(output, method, stepsize)
    }
}

genTestData = function() {
    testData = list()
    N = 10^3    # Number of obeservations
    testData$minibatchSize = 100
    # True locations
    theta = matrix(c(0, 0.5, 0, 0.5), ncol = 2)
    testData$truth = theta
    # Allocation probabilities
    z = sample(2, N, replace = TRUE)
    # Preallocate data
    testData$dataset = list("X" = matrix(rep(0, N*2), ncol = 2)) 
    # Simulate dataset
    for ( i in 1:N ) {
        theta_i = theta[z[i],]
        testData$dataset$X[i,] = mvrnorm(1, theta_i, diag(2))
    }
    # Sample random starting points
    testData$params = list("theta1" = rnorm(2), "theta2" = rnorm(2))
    testData$stepsizes = c(1e-3, 5e-4, 1e-4, 5e-5, 1e-5)
    testData$optStepsize = 5e-5
    testData$nIters = 2 * 10^4
    return(testData)
}

rmBurnIn = function(output, method, stepsize) {
    # Remove burn-in if non-cv, store chain of theta1
    theta = output$theta1
    if (method == "sghmc") {
        theta = theta[-c(1:2000),]
    } else if (method %in% c("sgld", "sgnht")) {
        theta = theta[-c(1:10000),]
    }
    write.table(theta, paste0("./gaussMix/", method), row.names = FALSE, col.names = FALSE)
}

# Use RSTAN to generate 'true sample' for comparison
hmc = function(testSeed) {
    message("\nSimulating Hamiltonian Monte Carlo using STAN to act as 'truth' for plot")
    # Generate test data given seed
    set.seed(testSeed)
    testData = genTestData()
    rstan_options(auto_write = TRUE)
    options(mc.cores = parallel::detectCores())
    # Setup input to STAN
    stanData = list()
    stanData$x = testData$dataset$X
    stanData$N = nrow(stanData$x)
    stanData$Sigma = diag(2)
    stanData$theta0 = c(0, 0)
    stanData$Sigma0 = 10 * diag(2)
    output = stan(file = "./gaussMix/gauss_mixture.stan", data = stanData, 
            iter = testData$nIters, chains = 1, seed = testSeed)
    # Extract output, convert to stacked matrix, store
    output = extract(output, permuted = FALSE)
    tempMat = matrix(rep(0, length(output[,,1]) * 4), ncol = 2)
    tempMat[,1] = c(output[,,1], output[,,3])
    tempMat[,2] = c(output[,,2], output[,,4])
    output = tempMat
    write.table(output, "./gaussMix/truth.dat", row.names = FALSE, col.names = FALSE)
}
