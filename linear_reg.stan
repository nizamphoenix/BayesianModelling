
// This Stan program defines a linear regression model


data {
  int<lower=0> n;   //number of observations
  int<lower=0> P;   //number of parameters
  real y[n];          //response vector
  matrix[n,P] X;     //design matrix for fixed effects (includes intercept)
}



// accepts two sets of parameters 'beta', and 'sigma'.
parameters {
  vector[P] beta; //vector of fixed effects of length P.
  real<lower=0> tau; //residual precision
}

transformed parameters {
  real<lower=0> sigma;
sigma = pow(tau, -0.5); //residual standard deviation
}


// 'y'is normal with mean X*beta and variance sigma.
// and assume a vague gamma prior for tau = 1/sigma^2.
model {
  y ~ normal(X*beta,sigma);  //likelihood
  tau ~ gamma(0.001,0.001);
}
