
// mixed model 

data {
  int<lower=0> n;   //number of observations
  int<lower=0> Q;   //number of random effect parameter
  int<lower=0> P;   //number of fixed effect parameters, If fitting ridge regression p =1.
  real y[n];          //response vector
  matrix[n,Q] Z;     //indicator matrix for random effect levels
  matrix[n,P] X;     //design matrix for fixed effects
}


// accepts four sets of parameters 'beta', 'u', 'sigma', 'sigmau'.
parameters {
  vector[P] beta; //vector of fixed effects of length P.
  vector[Q] u; //vector of random effects of length Q.
  real<lower=0> tau; //residual precision
  real<lower=0> tauu; //random effect precision
}

transformed parameters {
  real<lower=0> sigma;
  real<lower=0> sigmau;
sigma = pow(tau, -0.5); //residual standard deviation
sigmau = pow(tauu, -0.5); //random effect standard deviation
}



// 'y' to be normal with mean X*beta+ Z*u and standard deviation sigma.
// and assume a i.i.d. normal prior for u.
model {
  u ~ normal(0,sigmau);           //prior for random effects.
  tau ~ gamma(0.001,0.001);       //prior for residual precision
  tauu ~ gamma(0.001,0.001);      //prior for random effect precision
  y ~ normal(X*beta+ Z*u,sigma);  //likelihood
}
