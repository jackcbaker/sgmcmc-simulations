### Run simulations in Section 5.2 ###
library(sgmcmc)

# Declare log likelihood, as detailed in manuscript
logLik = function(params, dataset) {
    yEstimated = 1 / (1 + tf$exp( - tf$squeeze(params$bias + tf$matmul(dataset$X, params$beta))))
    logLik = tf$reduce_sum(dataset$y * tf$log(yEstimated) + (1 - dataset$y) * tf$log(1 - yEstimated))
    return(logLik)
}

# Declare log prior
logPrior = function(params) {
    logPrior = - (tf$reduce_sum(tf$abs(params$beta)) + tf$reduce_sum(tf$abs(params$bias)))
    return(logPrior)
}

runSimulations = function() {
    # Generate test data given seed
    testSeed = 1
    set.seed(testSeed)
    testData = genTestData()
    # Set base arguments for each sgmcmc method run in simulation, assign good stepsizes
    stepsizes = list("sgld" = 5e-6, "sghmc" = 5e-7, "sgnht" = 1e-7, "sgldcv" = 1e-5, 
            "sghmccv" = 1e-6, "sgnhtcv" = 1e-6)
    argsCurr = list( "logLik" = logLik, "dataset" = testData$dataset, "params" = testData$params,
            "logPrior" = logPrior, "minibatchSize" = testData$minibatchSize, 
            "verbose" = FALSE, "seed" = testSeed, nIters = 2 * 10^4 )
    # Run standard methods
    cat("method: ")
    for (method in c("sgld", "sghmc", "sgnht")) {
        cat(paste0(method, " "))
        # Account for fact that sghmc has 5x computational cost by reducing number of iterations
        if (method == "sghmc") {
            argsCurr$nIters = 4000
        } else {
            argsCurr$nIters = 2 * 10^4
        }
        argsCurr$stepsize = stepsizes[[method]]
        output = do.call(method, argsCurr)
        # Calculate log loss every 10 iterations and write to file (used for plotting)
        logLoss = calcLogLoss(testData$testset, output, method)
        write.table(logLoss, paste0("./logReg/", method))
    }
    # Run control variate methods after adding relevant arguments
    argsCurr$optStepsize = testData$optStepsize
    argsCurr$nIters = 10^4
    for (method in c("sgldcv", "sghmccv", "sgnhtcv")) {
        cat(paste0(method, " "))
        # Account for fact that sghmccv has 5x computational cost by reducing number of iterations
        if (method == "sghmccv") {
            argsCurr$nIters = 2000
        } else {
            argsCurr$nIters = 10^4
        }
        argsCurr$stepsize = stepsizes[[method]]
        output = do.call(method, argsCurr)
        # Calculate log loss every 10 iterations and write to file (used for plotting)
        logLoss = calcLogLoss(testData$testset, output, method)
        write.table(logLoss, paste0("./logReg/", method))
    }
    cat("\n")
}

genTestData = function() {
    testData = list()
    covertype = getDataset("covertype")
    # Remove random test set
    testObservations = sample(nrow(covertype), 10^4)
    testData$testset = covertype[testObservations,]
    # Separate response and data
    X = covertype[-c(testObservations),2:ncol(covertype)]
    y = covertype[-c(testObservations),1]
    # Declare dataset
    testData$dataset = list( "X" = X, "y" = y )
    # Get the dimension of X, needed to set shape of params$beta. Sample random start
    d = ncol(testData$dataset$X)
    testData$params = list( "bias" = rnorm(1), "beta" = matrix( rnorm(d), nrow = d ) )
    testData$stepsizes = c(1e-5, 5e-6, 1e-6, 5e-7, 1e-7)
    testData$optStepsize = 1e-6
    testData$minibatchSize = 500
    testData$nIters = 2 * 10^4
    return(testData)
}

calcLogLoss = function(testset, output, method) {
    # Remove necessary burn-in since methods are run for different numbers of iterations
    if (method == "sghmc") {
        output$bias = output$bias[-c(1:2000)]
        output$beta = output$beta[-c(1:2000),,]
        nIters = 2000
    } else if (nrow(output$beta) == 20000) {
        output$bias = output$bias[-c(1:10^4)]
        output$beta = output$beta[-c(1:10^4),,]
        nIters = 10^4
    } else {
        output$bias = drop(output$bias)
        output$beta = drop(output$beta)
        nIters = length(output$bias)
    }
    # Separate out response and explanatory variables in test set
    yTest = testset[,1]
    XTest = testset[,2:ncol(testset)]
    iterations = seq(from = 1, to = nIters, by = 10)
    logLoss = rep(0, length(iterations))
    # Calculate log loss every 10 iterations
    for ( iter in 1:length(iterations) ) {
        j = iterations[iter]
        # Get parameters at iteration j
        beta0_j = output$bias[j]
        beta_j = output$beta[j,]
        # Calculate log loss at each test set point
        for ( i in 1:length(yTest) ) {
            pihat_ij = 1 / (1 + exp(- beta0_j - sum(XTest[i,] * beta_j)))
            y_i = yTest[i]
            LogPred_curr = - (y_i * log(pihat_ij) + (1 - y_i) * log(1 - pihat_ij))
            logLoss[iter] = logLoss[iter] + 1 / length(yTest) * LogPred_curr
        }
    }
    return(logLoss)
}
