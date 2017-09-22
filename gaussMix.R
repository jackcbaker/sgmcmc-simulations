library(ggplot2)
library(sgmcmc)
library(MASS)

runSimulations = function() {
    stepsizeList = list("sgld" = 5e-3, "sghmc" = 5e-4, "sgnht" = 3e-4, "sgldcv" = 1e-2, "sghmccv" = 1.5e-3, "sgnhtcv" = 3e-3)
    methods = c("sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv")
    plotList = list()
    for (method in methods) {
        cat(paste0(method, " "))
        stepsize = stepsizeList[[method]]
        runSeed(2, method, stepsize)
    }
    cat("\n")
}

plotGM = function() {
    methods = c( "sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv" )
    truth = read.table("./gaussMix/truth.dat")[1:10000,]
    colnames(truth) = c("dim1", "dim2")
    truth$Group = "truth"
    plotList = list()
    for (method in methods) {
        current = read.table(paste0("gaussMix/", method))
        colnames(current) = c("dim1", "dim2")
        current$Method = method
        current$Group = method
        truthCurr = truth
        truthCurr$Method = method
        current$Method = method    
        truthCurr$Method = method
        plotList[[method]] = rbind(current, truthCurr)
    }

    plotFrame = do.call("rbind", plotList)
    # Make plot clearer by ordering
    methods = c("truth", "sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv")
    colors = c("#000000", "#E69F00", "#56B4E9", "#009E73", "#E69F00", "#56B4E9", "#009E73")
    plotFrame$Group = factor(plotFrame$Group, levels = methods)
    plotFrame$Method = factor(plotFrame$Method, levels = methods[-1])
    p = ggplot(plotFrame, aes(x = dim1, y = dim2, color = Group)) +
        stat_density2d(alpha = 0.7, size = 1) +
        scale_color_manual(values=colors, name = methods) +
        facet_grid(.~Method)
    ggsave("./plots/sim-gm.pdf", width = 12, height = 3)
}

# Declare log likelihood
logLik = function( params, dataset ) {
    # Declare Sigma (assumed known)
    SigmaDiag = c(1, 1)
    # Declare distribution of each component
    component1 = tf$contrib$distributions$MultivariateNormalDiag( params$theta1, SigmaDiag )
    component2 = tf$contrib$distributions$MultivariateNormalDiag( params$theta2, SigmaDiag )
    # Declare allocation probabilities of each component
    probs = tf$contrib$distributions$Categorical(c(0.5,0.5))
    # Declare full mixture distribution given components and allocation probabilities
    distn = tf$contrib$distributions$Mixture(probs, list(component1, component2))
    # Declare log likelihood
    logLik = tf$reduce_sum( distn$log_prob(dataset$X) )
    return( logLik )
}

# Declare log prior ( theta_ij ~ N( 0, 10 ) )
logPrior = function( params ) {
    # Declare hyperparameters mu0 and Sigma0
    mu0 = c( 0, 0 )
    Sigma0Diag = c(10, 10)
    # Declare prior distribution
    priorDistn = tf$contrib$distributions$MultivariateNormalDiag( mu0, Sigma0Diag )
    # Declare log prior density and return
    logPrior = priorDistn$log_prob( params$theta1 ) + priorDistn$log_prob( params$theta2 )
    return( logPrior )
}

genTestData = function() {
    testData = list()
    N = 10^3    # Number of obeservations
    testData$minibatchSize = 100
    # Generate test data
    # True locations (0, 0) and (0.1, 0.1)
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

runSeed = function(testSeed, method, stepsize) {
    # Generate test data
    set.seed(testSeed)
    testData = genTestData()
    argsCurr = list( "logLik" = logLik, "dataset" = testData$dataset, "params" = testData$params,
            "logPrior" = logPrior, "minibatchSize" = testData$minibatchSize, nIters = 2 * 10^4, 
            "verbose" = FALSE, "seed" = testSeed, "stepsize" = stepsize )
    if (grepl("cv", method)) {
        # Set additional arguments for sgldcv
        argsCurr$optStepsize = testData$optStepsize
        if (method == "sghmccv") {
            argsCurr$nIters = 2000
        } else {
            argsCurr$nIters = 10^4
        }
        output = do.call(method, argsCurr)
        rmBurnIn(output, method, stepsize)
    } else {
        if (method == "sghmc") {
            argsCurr$nIters = 4000
        } else {
            argsCurr$nIters = 2 * 10^4
        }
        output = do.call(method, argsCurr)
        rmBurnIn(output, method, stepsize)
    }
}

rmBurnIn = function(output, method, stepsize) {
    # Remove burn-in if non-cv
    theta = output$theta1
    if (method == "sghmc") {
        theta = theta[-c(1:2000),]
    } else if (method %in% c("sgld", "sgnht")) {
        theta = theta[-c(1:10000),]
    }
    write.table(theta, paste0("./gaussMix/", method), row.names = FALSE, col.names = FALSE)
}
