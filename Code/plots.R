library(ggplot2)

plotStanSGMCMC = function() {
    Nseq = c(10^3, 5000, 10^4, 5 * 10^4, 10^5, 5*10^5, 10^6)
    stan = list()
    sgmcmc = list()
    for (N in Nseq) {
        kl = read.table(paste0("stan-sgmcmc/stan/", N, "/kl"))$V1
        df = data.frame("kl" = mean(kl), "klmin" = min(kl), "klmax" = max(kl))
        stan[[N]] = df
        stan[[N]]$N = N
        stantime = read.table(paste0("stan-sgmcmc/stan/", N, "/time"))$V1
        stan[[N]]$time = mean(stantime)
        stan[[N]]$timemin = min(stantime)
        stan[[N]]$timemax = max(stantime)
        kl = read.table(paste0("stan-sgmcmc/sgmcmc/", N, "/kl"))$V1
        df = data.frame("kl" = mean(kl), "klmin" = min(kl), "klmax" = max(kl))
        sgmcmc[[N]] = df
        sgmcmc[[N]]$N = N
        sgmcmctime = read.table(paste0("stan-sgmcmc/sgmcmc/", N, "/time"))$V1
        sgmcmc[[N]]$time = mean(sgmcmctime)
        sgmcmc[[N]]$timemin = min(sgmcmctime)
        sgmcmc[[N]]$timemax = max(sgmcmctime)
    }
    standf = do.call(rbind, stan)
    standf$Method = "Stan"
    sgmcmcdf = do.call(rbind, sgmcmc)
    sgmcmcdf$Method = "sgmcmc"

    plotFrame = rbind(standf, sgmcmcdf)
    plotFrame$N = as.numeric(plotFrame$N)
    p = ggplot(plotFrame, aes(x = N, y = kl, ymin = klmin, ymax = klmax, fill = Method)) +
        geom_ribbon(alpha = 0.5) +
        xlab("Number of Observations") +
        scale_x_log10(breaks=c(10^3, 10^4, 10^5, 10^6),labels=c("10^3", "10^4", "10^5", "10^6")) +
        ylab("KL divergence")
    ggsave("../Figures/sgmcmc-stan-kl.pdf", width = 4, height = 2)

    p = ggplot(plotFrame, aes(x = N, y = time, ymin = timemin, ymax = timemax, fill = Method)) +
        geom_ribbon(alpha = 0.5) +
        xlab("Number of Observations") +
        scale_x_log10(breaks=c(10^3, 10^4, 10^5, 10^6),labels=c("10^3", "10^4", "10^5", "10^6")) +
        ylab("Run time (secs)")
    ggsave("../Figures/sgmcmc-stan-time.pdf", width = 4, height = 2)
}

plotGM = function() {
    methods = c( "sgld", "sghmc", "sgnht", "sgldcv", "sghmccv", "sgnhtcv" )
    # Read in truth baseline calculated using STAN, take theta1 output only
    truth = read.table("./gaussMix/truth.dat")[1:10000,]
    colnames(truth) = c("dim1", "dim2")
    truth$Group = "truth"
    # Add output from gaussMix.R to one big data frame for plotting
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
    ggsave("../Figures/sim-gm.pdf", width = 12, height = 3)
}

plotLogReg = function() {
    # Add log loss output from logReg.R to one big data frame for plotting
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
    ggsave("../Figures/sim-lr.pdf", width = 7, height = 3)
}

plotNN = function() {
    # Add log loss output from nn.R to one big data frame for plotting
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
    # Replace iteration with data processed
    plotFrame$processed = plotFrame$Iteration * 500 / 55000
    # Make plot clearer by ordering
    colorPal = c("#E69F00", "#56B4E9", "#009E73", "#E69F00", "#56B4E9", "#009E73")
    plotFrame$Method = factor(plotFrame$Method, levels = methods)
    p = ggplot(plotFrame, aes(x = processed, y = V1, color = Method)) +
        geom_line(alpha = 0.8) +
        ylab("Log loss of test set") +
        xlab("Proportion of dataset processed") +
        scale_color_manual(values=colorPal, name = methods) +
        facet_grid(. ~ Type)
    ggsave("../Figures/sim-nn.pdf", width = 7, height = 3)
}
