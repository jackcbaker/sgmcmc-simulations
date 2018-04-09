library(sgmcmc)
library(rstan)

runSimulations = function() {
    stanNorm()
    sgmcmcNorm()
}

stanNorm = function() {
    Nseq = c(10^3, 5000, 10^4, 5 * 10^4, 10^5, 5*10^5, 10^6)
    # Simulate data
    set.seed(2)
    fullset = list("x" = rnorm(10^6))
    # Initial run of Stan as a check
    dataset = list()
    dataset$x = fullset$x[1:100]
    dataset$N = 100
    stanfile = "stan-sgmcmc/norm.stan"
    fit = stan(file = stanfile, data = dataset, iter = 2000, chains = 1, init = 0)
    # Sample from examples with differing number of observations
    for (N in Nseq) {
        set.seed(2)
        dataset = list()
        dataset$x = fullset$x[1:N]
        dataset$N = N
        # Allocate vectors to store kl and runtime results
        runtime = rep(NA, 5)
        kl = rep(NA, 5)
        # Average over 5 seeds
        for (seed in 1:5) {
            set.seed(seed)
            # Sample from model using STAN, measure runtime including setup
            start = proc.time()
            fit = stan(file = stanfile, data = dataset, iter = 2000, chains = 1, seed = seed, init = 0)
            print(unname((proc.time() - start)))
            runtime[seed] = unname((proc.time() - start))[3]
            theta = drop(extract(fit, "theta", permuted = FALSE))
            kl[seed] = klDiv(dataset$x, theta)
        }
        # Store runtime and kl results
        outdir = paste0("stan-sgmcmc/stan/", N, "/")
        dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
        fname = paste0(outdir, "time")
        write.table(runtime, fname, row.names = FALSE, col.names = FALSE)
        fname = paste0(outdir, "kl")
        write.table(kl, fname, row.names = FALSE, col.names = FALSE)
    }
}

# Declare logLik and logPrior for sgmcmc
# Standard normal density with N(0,10) prior
logLik = function(params, dataset) {
    distn = tf$distributions$Normal(params$theta, 1.0)
    logLik = tf$reduce_sum(distn$log_prob(dataset$x))
    return(logLik)
}

logPrior = function(params) {
    distn = tf$distributions$Normal(0.0, 10.0)
    logPrior = distn$log_prob(params$theta)
    return(logPrior)
}

sgmcmcNorm = function() {
    Nseq = c(10^3, 5000, 10^4, 5*10^4, 10^5, 5 * 10^5, 10^6)
    # Simulate data
    set.seed(2)
    fullset = list("x" = rnorm(10^6))
    stepsizes = list("1000" = 3e-5, "5000" = 1e-5, "10000" = 7e-6, "50000" = 5e-6, "1e+05" = 3e-6, "5e+05" = 1e-6, "1e+06" = 3e-7)
    dataset = list()
    dataset$x = fullset$x[1:100]
    params = list("theta" = 0)
    # Initial run of TensorFlow as check
    stepsize = 8e-4
    output = sgld(logLik, dataset, params, stepsize, logPrior = logPrior, minibatchSize = 10, nIters = 2000, verbose = FALSE) 
    # Fit models with different numbers of observations
    for (N in Nseq) {
        # Set tuning constants and current dataset
        stepsize = stepsizes[[as.character(N)]]
        dataset = list()
        dataset$x = fullset$x[1:N]
        d = ncol(dataset$X)
        params = list("theta" = 0)
        # Allocate vectors to store kl and runtime results
        runtime = rep(NA, 5)
        kl = rep(NA, 5)
        for (seed in 1:5) {
            set.seed(seed)
            # Measure run time including setup costs
            start = proc.time()
            # When N == 10^6 a slightly higher minibatch size seems needed to
            # keep a reasonable KL divergence (this will show up in impact on runtime)
            if (N < 5 * 10^5) {
                output = sgldcv(logLik, dataset, params, stepsize, stepsize, logPrior = logPrior, minibatchSize = 10^3, nIters = 1000, nItersOpt = 1000, verbose = FALSE, seed = seed) 
            } else {
                output = sgldcv(logLik, dataset, params, stepsize, stepsize, logPrior = logPrior, minibatchSize = 5 * 10^3, nIters = 1000, nItersOpt = 1000, verbose = FALSE, seed = seed) 
            }
            runtime[seed] = unname((proc.time() - start))[3]
            theta = output$theta
            kl[seed] = klDiv(dataset$x, theta)
        }
        # Store KL and runtime results
        outdir = paste0("stan-sgmcmc/sgmcmc/", N, "/")
        dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
        fname = paste0(outdir, "time")
        write.table(runtime, fname, col.names = FALSE, row.names = FALSE)
        fname = paste0(outdir, "kl")
        write.table(kl, fname, row.names = FALSE, col.names = FALSE)
    }
}

# Calculate KL divergence from truth
klDiv = function(x, theta) {
    # Calculate true posterior from data
    N = length(x)
    sigma = 1 / (1 / 10.0 + N)
    mean = sigma * sum(x)
    # Calculate estimated mean and variance
    meanapprox = mean(theta)
    sigmaapprox = var(theta)
    # Calculate KL-divergence
    kl = log(sqrt(sigmaapprox / sigma)) + (sigma + (mean - meanapprox)^2) / (2 * sigmaapprox) - 1 / 2
    return(kl)
}
