data {
    int N;
    real x[N];
}
parameters {
    real theta;
}
model {
    theta ~ normal(0, 10);
    x ~ normal(theta, 1);
}
