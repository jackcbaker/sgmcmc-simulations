### Run code in Section 4 ###
library(sgmcmc) 
library(ggplot2)

# Declare log likelihood for logistic regression as in manuscript
logLik = function(params, dataset) { 
    yEst = 1 / (1 + tf$exp( - tf$squeeze(params$bias +
            tf$matmul(dataset$X, params$beta))))
    logLik = tf$reduce_sum(dataset$y * tf$log(yEst) +
            (1 - dataset$y) * tf$log(1 - yEst))
    return(logLik)
}

# Declare log prior for logistic regression as in manuscript
logPrior = function(params) { 
    logPrior = - (tf$reduce_sum(tf$abs(params$beta)) + 
            tf$reduce_sum(tf$abs(params$bias))) 
    return(logPrior) 
}

# This function will run a word for word copy of Section 4 'Usage' in the manuscript
# (minus any print/output statements)
runSimulations = function() {
    # Download dataset
    covertype = getDataset("covertype") 
    # Run code from 'standard usage' Section 4.1 in manuscript
    cat("example running: ")
    standardUsage(covertype)
    # Run code from 'storage constraints' Section 4.2 in manuscript
    stepByStepUsage(covertype)
}

standardUsage = function(covertype) {
    # Split the data into predictors and response 
    X = covertype[,2:ncol(covertype)] 
    y = covertype[,1] 
    # Create dataset list for input 
    dataset = list( "X" = X, "y" = y )
    # Get the dimension of X, needed to set shape of params
    d = ncol(dataset$X)
    params = list( "bias" = 0, "beta" = matrix( rep( 0, d ), nrow = d ) )
    stepsize = list("beta" = 2e-5, "bias" = 2e-5)
    cat("sgld ")
    output = sgld( logLik, dataset, params, stepsize, 
            logPrior = logPrior, minibatchSize = 500, nIters = 10000, verbose = FALSE, seed = 13 )
    # Get test set to calculate log loss
    set.seed(13) 
    testSample = sample(nrow(dataset$X), 10^4) 
    testset = list( "X" = dataset$X[testSample,], "y" = dataset$y[testSample] ) 
    dataset = list( "X" = dataset$X[-testSample,], "y" = dataset$y[-testSample] ) 
    cat("sgldcv ")
    output = sgldcv( logLik, dataset, params, 5e-6, 5e-6,  
            logPrior = logPrior, minibatchSize = 500, nIters = 11000, seed = 13, verbose = FALSE ) 
    # Remove burn-in 
    output$beta = output$beta[-c(1:1000),,] 
    output$bias = output$bias[-c(1:1000)]
    iterations = seq(from = 1, to = 10^4, by = 10) 
    logLoss = rep(0, length(iterations)) 
    # Calculate log loss every 10 iterations 
    for ( iter in 1:length(iterations) ) { 
        j = iterations[iter] 
        beta0_j = output$bias[j] 
        beta_j = output$beta[j,] 
        for ( i in 1:length(testset$y) ) { 
            piCurr = 1 / (1 + exp(- beta0_j - sum(testset$X[i,] * beta_j))) 
            y_i = testset$y[i] 
            logLossCurr = - ( (y_i * log(piCurr) + (1 - y_i) * log(1 - piCurr)) ) 
            logLoss[iter] = logLoss[iter] + 1 / length(testset$y) * logLossCurr 
        } 
    }
    # Plot output
    plotFrame = data.frame("Iteration" = iterations, "logLoss" = logLoss)
    ggplot(plotFrame, aes(x = Iteration, y = logLoss)) +
        geom_line(color = "maroon") +
        ylab("Log loss of test set")
    ggsave("plots/usage.pdf", width = 7, height = 3)
}

stepByStepUsage = function(covertype) {
    cat("step-by-step\n")
    # Split the data into predictors and response 
    X = covertype[,2:ncol(covertype)] 
    y = covertype[,1] 
    # Create dataset list for input 
    dataset = list( "X" = X, "y" = y )
    # Get test set to calculate log loss
    set.seed(13) 
    testSample = sample(nrow(dataset$X), 10^4) 
    testset = list( "X" = dataset$X[testSample,], "y" = dataset$y[testSample] ) 
    dataset = list( "X" = dataset$X[-testSample,], "y" = dataset$y[-testSample] ) 
    # Get the dimension of X, needed to set shape of params
    d = ncol(dataset$X)
    params = list( "bias" = 0, "beta" = matrix( rep( 0, d ), nrow = d ) )
    stepsize = list("beta" = 2e-5, "bias" = 2e-5)
    sgld = sgldSetup(logLik, dataset, params, stepsize, 
            logPrior = logPrior, minibatchSize = 500, seed = 13)
    testPlaceholder = list() 
    testPlaceholder[["X"]] = tf$placeholder(tf$float32, dim(testset[["X"]])) 
    testPlaceholder[["y"]] = tf$placeholder(tf$float32, dim(testset[["y"]])) 
    testSize = as.double(nrow(testset[["X"]])) 
    logLoss = - logLik(sgld$params, testPlaceholder) / testSize
    sess = initSess(sgld) 
    # Fill a feed dict with full test set (used to calculate log loss) 
    feedDict = dict() 
    feedDict[[testPlaceholder[["X"]]]] = testset[["X"]] 
    feedDict[[testPlaceholder[["y"]]]] = testset[["y"]] 
    # Burn-in chain 
    for (i in 1:10^4) { 
    # Print progress 
        if (i %% 100 == 0) { 
            progress = sess$run(logLoss, feed_dict = feedDict) 
        } 
        sgmcmcStep(sgld, sess) 
    } 
    # Initialise posterior mean estimate using value after burn-in 
    postMean = getParams(sgld, sess) 
    logLossOut = rep(0, 10^4 / 100) 
    # Run chain 
    for (i in 1:10^4) { 
        sgmcmcStep(sgld, sess) 
        # Update posterior mean estimate 
        currentState = getParams(sgld, sess) 
        for (paramName in names(postMean)) { 
            postMean[[paramName]] = (postMean[[paramName]] * i +  
                    currentState[[paramName]]) / (i + 1) 
        } 
        # Print and store log loss 
        if (i %% 100 == 0) { 
            logLossOut[i/100] = sess$run(logLoss, feed_dict = feedDict) 
        }
    }
}
