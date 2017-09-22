library(ggplot2)
library(sgmcmc)

runSimulations = function() {
    stepsizes = list("sgld" = 1e-4, "sghmc" = 5e-5, "sgnht" = 5e-6, "sgldcv" = 5e-5,
            "sghmccv" = 5e-6, "sgnhtcv" = 5e-7)
    for (method in names(stepsizes)) {
        cat(paste0(method, " "))
        stepsize = stepsizes[[method]]
        runSeed(1, method, stepsize)
    }
}

plotNN = function() {
    # Build a list of log loss plots
    methods = c("sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv")
    plotList = list()
    for (method in methods) {
        dataCurr = read.table(paste0("nn/", method))
        if(substr(method, nchar(method) - 1, nchar(method)) == "cv") {
            dataCurr$Type = "Control Variate"
        } else {
            dataCurr$Type = "Standard"
        }
        if (grepl("sghmc", method)) {
            dataCurr$Iteration = seq(from = 10, to = 10^4, by = 50)
        } else {
            dataCurr$Iteration = seq(from = 10, to = 10^4, by = 10)
        }
        dataCurr$Method = method
        plotList[[method]] = dataCurr
    }

    plotFrame = do.call("rbind", plotList)
    plotFrame$processed = plotFrame$Iteration * 500 / 55000
    # Set colors for plotting
    colorPal = c("#E69F00", "#56B4E9", "#009E73", "#E69F00", "#56B4E9", "#009E73")
    plotFrame$Method = factor(plotFrame$Method, levels = methods)

    p = ggplot(plotFrame, aes(x = processed, y = V1, color = Method)) +
        geom_line(alpha = 0.8) +
        ylab("Log loss of test set") +
        xlab("Proportion of dataset processed") +
        scale_color_manual(values=colorPal, name = methods) +
        facet_grid(. ~ Type)
    ggsave("plots/sim-nn.pdf", width = 7, height = 3)
}

logLik = function(params, dataset) {
    YEst = tf$nn$softmax(tf$matmul(dataset$X, params$B) + params$b)
    YEst = tf$nn$softmax(tf$matmul(YEst, params$A) + params$a)
    logLik = tf$reduce_sum(dataset$y * tf$log(YEst))
    return(logLik)
}

logPrior = function(params) {
    distLambda = tf$contrib$distributions$Gamma(1, 1)
    distA = tf$contrib$distributions$Normal(0, tf$rsqrt(params$lambdaA))
    logPriorA = tf$reduce_sum(distA$log_prob(params$A)) + distLambda$log_prob(params$lambdaA)
    distB = tf$contrib$distributions$Normal(0, tf$rsqrt(params$lambdaB))
    logPriorB = tf$reduce_sum(distB$log_prob(params$B)) + distLambda$log_prob(params$lambdaB)
    dista = tf$contrib$distributions$Normal(0, tf$rsqrt(params$lambdaa))
    logPriora = tf$reduce_sum(dista$log_prob(params$a)) + distLambda$log_prob(params$lambdaa)
    distb = tf$contrib$distributions$Normal(0, tf$rsqrt(params$lambdab))
    logPriorb = tf$reduce_sum(distb$log_prob(params$b)) + distLambda$log_prob(params$lambdab)
    logPrior = logPriorA + logPriorB + logPriora + logPriorb
    return(logPrior)
}

genTestData = function() {
    testData = list()
    # Load MNIST dataset
    data("mnist")
    # Declare dataset
    testData$dataset = list("X" = mnist$train$images, "y" = mnist$train$labels)
    testData$testset = list("X" = mnist$test$images, "y" = mnist$test$labels)
    # Sample initial weights from standard Normal
    d = ncol(testData$dataset$X)
    testData$params = list( "A" = matrix( rnorm(10*100), ncol = 10 ) )
    testData$params$B = matrix(rnorm(d*100), ncol = 100)
    # Sample initial bias parameters from standard Normal
    testData$params$a = rnorm(10)
    testData$params$b = rnorm(100)
    # Sample initial precision parameters from standard Gamma
    testData$params$lambdaA = rgamma(1, 1)
    testData$params$lambdaB = rgamma(1, 1)
    testData$params$lambdaa = rgamma(1, 1)
    testData$params$lambdab = rgamma(1, 1)
    testData$stepsizes = c(1e-5, 5e-6, 1e-6, 5e-7, 1e-7)
    testData$optStepsize = 3e-5
    testData$minibatchSize = 500
    testData$nIters = 2 * 10^4
    testData$testSize = nrow(testData$testset$X)
    testData$testPlaceholders = list()
    for (pname in names(testData$dataset)) {
        testData$testPlaceholders[[pname]] = tf$placeholder(tf$float32, c(testData$testSize, ncol(testData$dataset[[pname]])) )
    }
    # Create test dictionary full of test data to feed to testPlaceholders
    testData$testDict = dict()
    testData$testDict[[testData$testPlaceholders[["X"]]]] = testData$testset[["X"]]
    testData$testDict[[testData$testPlaceholders[["y"]]]] = testData$testset[["y"]]
    return(testData)
}

runSeed = function(testSeed, method, stepsize) {
    # Generate test data
    set.seed(testSeed)
    testData = genTestData()
    setupArgs = list( "logLik" = logLik, "dataset" = testData$dataset, 
            "params" = testData$params, "stepsize" = stepsize, "logPrior" = logPrior, 
            "minibatchSize" = testData$minibatchSize, seed = testSeed)
    sgmcmc = setupSGMCMC(method, setupArgs)
    logLoss = runSim(sgmcmc, testData)
    write.table(logLoss, paste0("nn/", method), row.names = FALSE, col.names = FALSE)
}

setupSGMCMC = function(method, setupArgs) {
    suffix = substr(method, nchar(method) - 1, nchar(method))
    if (suffix == "cv") {
        setupArgs$optStepsize = 3e-5
    }
    sgmcmc = do.call(paste0(method, "Setup"), setupArgs)
    return(sgmcmc)
}

runSim = function(sgmcmc, testData) UseMethod("runSim")

runSim.sgmcmccv = function(sgmcmc, testData) {
    testLogLoss = - 1 / as.double(testData$testSize) * logLik(sgmcmc$params, testData$testPlaceholders)
    sess = initSess(sgmcmc)
    # Account for 5x comp cost of sghmc by running for less long
    if (class(sgmcmc)[1] == "sghmc") {
        nIters = 2000
    } else {
        nIters = 10^4
    }
    logLossOut = rep(0, nIters / 10)
    # Run full SGMCMC
    for (i in 1:nIters) {
        sgmcmcStep(sgmcmc, sess)
        if (i %% 10 == 0) {
            logLossOut[i / 10] = sess$run(testLogLoss, feed_dict = testData$testDict)
        }
    }
    return(logLossOut)
}

runSim.sgmcmc = function(sgmcmc, testData) {
    testLogLoss = - 1 / as.double(testData$testSize) * logLik(sgmcmc$params, testData$testPlaceholders)
    sess = initSess(sgmcmc)
    # Account for 5x comp cost of sghmc by running for less long
    if (class(sgmcmc)[1] == "sghmc") {
        nIters = 2000
    } else {
        nIters = 10^4
    }
    # Burn-in
    for (i in 1:nIters) {
        sgmcmcStep(sgmcmc, sess)
        if (i %% 10 == 0) {
#            print(sess$run(testLogLoss, feed_dict = testData$testDict))
        }
    }
    # Run full SGMCMC
    logLossOut = rep(0, nIters / 10)
    for (i in 1:nIters) {
        sgmcmcStep(sgmcmc, sess)
        if (i %% 10 == 0) {
            logLossOut[i / 10] = sess$run(testLogLoss, feed_dict = testData$testDict)
        }
    }
    return(logLossOut)
}
