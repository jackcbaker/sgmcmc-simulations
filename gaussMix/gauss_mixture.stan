data {
    int<lower=1> N;         // number of data points
    row_vector[2] x[N];     // observations
    matrix[2,2] Sigma;   // scales of mixture components (common)
    vector[2] theta0;          // hyperparameters for location
    matrix[2,2] Sigma0;  // Scale hyperparameter for location
}
parameters {
    row_vector[2] theta1;    // locations of mixture components
    row_vector[2] theta2;    // locations of mixture components
}
model {
    real ps[2];             // temp for log component densities
    theta1 ~ multi_normal( theta0, Sigma0 );
    theta2 ~ multi_normal( theta0, Sigma0 );
    for (n in 1:N) {
        ps[1] = log(1.0/2) + multi_normal_lpdf(x[n]|theta1,Sigma);
        ps[2] = log(1.0/2) + multi_normal_lpdf(x[n]|theta2,Sigma);
        target += log_sum_exp(ps);
    }
}
