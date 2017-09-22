library(ggplot2)
library(sgmcmc)

runSimulations = function() {
    # Generate test data
    testSeed = 1
    set.seed(testSeed)
    testData = genTestData()
    stepsizes = list("sgld" = 5e-6, "sghmc" = 5e-7, "sgnht" = 1e-7, "sgldcv" = 1e-5, 
            "sghmccv" = 1e-6, "sgnhtcv" = 1e-6)
    argsCurr = list( "logLik" = logLik, "dataset" = testData$dataset, "params" = testData$params,
            "logPrior" = logPrior, "minibatchSize" = testData$minibatchSize, 
            "verbose" = FALSE, "seed" = testSeed, nIters = 2 * 10^4 )
    # Run standard methods
    for (method in c("sgld", "sghmc", "sgnht")) {
        cat(paste0(method, " "))
        if (method == "sghmc") {
            argsCurr$nIters = 4000
        } else {
            argsCurr$nIters = 2 * 10^4
        }
        argsCurr$stepsize = stepsizes[[method]]
        output = do.call(method, argsCurr)
        logLoss = calcLogPred(testData$testset, output, method)
        write.table(logLoss, paste0("./logReg/", method))
    }
    # Add arguments for control variates
    argsCurr$optStepsize = testData$optStepsize
    argsCurr$nIters = 10^4
    for (method in c("sgldcv", "sghmccv", "sgnhtcv")) {
        cat(paste0(method, " "))
        if (method == "sghmccv") {
            argsCurr$nIters = 2000
        } else {
            argsCurr$nIters = 10^4
        }
        argsCurr$stepsize = stepsizes[[method]]
        output = do.call(method, argsCurr)
        logLoss = calcLogPred(testData$testset, output, method)
        write.table(logLoss, paste0("./logReg/", method))
    }
    cat("\n")
}

plotLogReg = function() {
    plotList = list()
    for (method in dir("logReg")) {
        plotList[[method]] = read.table(paste0("logReg/",method))
        if (grepl("sghmc", method)) {
            plotList[[method]]$Iteration = seq(from = 10, to = 10^4, by = 50)
        } else {
            plotList[[method]]$Iteration = seq(from = 10, to = 10^4, by = 10)
        }
        plotList[[method]]$Method = method
        suffix = substr(method, nchar(method) - 1, nchar(method))
        if (suffix == "cv") {
            plotList[[method]]$Type = "Control variate"
        } else {
            plotList[[method]]$Type = "Standard"
        }
    }
    plotFrame = do.call("rbind", plotList)
    # Replace iteration with data processed
    plotFrame$process = plotFrame$Iteration * 500 / 571012
    # Make plot clearer by ordering
    methods = c("sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv")
    colors = c("#E69F00", "#56B4E9", "#009E73", "#E69F00", "#56B4E9", "#009E73")
    plotFrame$Method = factor(plotFrame$Method, levels = methods)
    p = ggplot(plotFrame, aes(x = process, y = x, color = Method)) +
        geom_line(alpha = 0.8) +
        ylab("Log loss of test set") +
        xlab("Proportion of dataset processed") +
        scale_color_manual(values=colors, name = methods) +
        facet_grid(. ~ Type)
    ggsave("plots/sim-lr.pdf", width = 7, height = 3)
}

logLik = function(params, dataset) {
    yEstimated = 1 / (1 + tf$exp( - tf$squeeze(params$bias + tf$matmul(dataset$X, params$beta))))
    logLik = tf$reduce_sum(dataset$y * tf$log(yEstimated) + (1 - dataset$y) * tf$log(1 - yEstimated))
    return(logLik)
}

logPrior = function(params) {
    logPrior = - (tf$reduce_sum(tf$abs(params$beta)) + tf$reduce_sum(tf$abs(params$bias)))
    return(logPrior)
}

genTestData = function() {
    testData = list()
    data(covertype)
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

calcLogPred = function(testset, output, method) {
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
    yTest = testset[,1]
    XTest = testset[,2:ncol(testset)]
    iterations = seq(from = 1, to = nIters, by = 10)
    avLogPred = rep(0, length(iterations))
    # Calculate log predictive every 10 iterations
    for ( iter in 1:length(iterations) ) {
        j = iterations[iter]
        # Get parameters at iteration j
        beta0_j = output$bias[j]
        beta_j = output$beta[j,]
        for ( i in 1:length(yTest) ) {
            pihat_ij = 1 / (1 + exp(- beta0_j - sum(XTest[i,] * beta_j)))
            y_i = yTest[i]
            # Calculate log predictive at current test set point
            LogPred_curr = - (y_i * log(pihat_ij) + (1 - y_i) * log(1 - pihat_ij))
            avLogPred[iter] = avLogPred[iter] + 1 / length(yTest) * LogPred_curr
        }
    }
    return(avLogPred)
}
