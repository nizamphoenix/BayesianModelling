class State(pm.Categorical):
    def __init__(self, trans_prob=None, init_prob=None, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.trans_prob = trans_prob
        self.init_prob = init_prob
        self.mode = tt.cast(0,dtype='int64')
        self.k = 2

    def logp(self, x):
        trans_prob = self.trans_prob
        p = trans_prob[x[:-1]] # probability of transitioning based on previous state
        x_i = x[1:]            # the state you end up in
        log_p = pm.Categorical.dist(p, shape=(self.shape[0],2)).logp_sum(x_i)
        return pm.Categorical.dist(self.init_prob).logp(x[0]) + log_p
