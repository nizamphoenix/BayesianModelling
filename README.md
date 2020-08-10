### A repository of bayesian modeling and Bayesian Neural Networks


Bayesian modeling unlocks narrative explanations of data, hence the efficacy to drive decision making.  
Bayesian Neural Networks(BNN) incorporate uncertainty in predictions, mitigating overfitting and making over-confident predictions in conventional neural networks.  

- BNNs model the distribution of each parameter of the network that is being trained by treating each parameter as a random variable rather as an unknown constant that is ultimately estimated at the end of training. So, technically the mean and standard deviation(if applicable) of evry parameter is learned during training of the bayesian networks.  
- In BNNs,the loss function is not just a cross-entropy function for a classification problem but a KL-divergence loss added to the cross-entropy value since a distribution is being learned. ELBo(Evidence lower bound) is used to learn the marginal likelihood of the data.
