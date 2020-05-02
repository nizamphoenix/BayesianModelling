
// This Stan program defines a LASSO regression.

  int<lower=0> n;   //number of observations
  int<lower=0> Q;   //number of parameters where LASSO penalty is applied.
  int<lower=0> P;   //number of fixed effect levels.
  real<lower=0> lambdainv;   //inverse of LASSO lambda. Note this is the inverse due to the  //parameterisation of the density in Stan.
  real y[n];      //response vector
  matrix[n,Q] Z;     //indicator matrix for random effect levels
  matrix[n,P] X;     //design matrix for fixed effects
}

// The parameters accepted by the model. 
// accepts three sets of parameters 'beta', 'u' and 'sigma'.
parameters {
  vector[P] beta; //vector of fixed effects of length P.
  vector[Q] u; //vector of random effects of length Q.
  real<lower=0> tau; //residual precision.
}

transformed parameters {
  real<lower=0> sigma;
sigma = pow(tau, -0.5); //residual standard deviation
}


// 'y' is be normal with mean X*beta+ Z*u and standard deviation sigma.
// and assume a i.i.d. double exponential prior for u.
model {
  u ~ double_exponential(0, lambdainv); //prior for effects with LASSO penalty.
  tau ~ gamma(0.001,0.001);        //prior for residual precision.
  y ~ normal(X*beta+ Z*u,sigma);  //likelihood.
}
