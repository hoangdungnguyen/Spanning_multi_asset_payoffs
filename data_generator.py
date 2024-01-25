import numpy as np
import torch
from scipy.stats import random_correlation

# DataGenerator

class DataGenerator:
    def __init__(self, payoff_dict, payoff, input_dim, strike = 1.):
        self.strike = strike
        self.input_dim = input_dim
        self.pay_func = payoff_dict[payoff]['pay_func']
        self.lb = payoff_dict[payoff]['lb'+str(int(input_dim))]
        self.ub = payoff_dict[payoff]['ub'+str(int(input_dim))]
        
        # configuration for log normal sampling
        self.r = 0.02
        self.T = 1
        min_vol = 0.1
        max_vol = 0.7
        
        try:
            eig_2 = abs(np.random.RandomState(0).randn(input_dim))
            eig_2 = eig_2/eig_2.sum()*self.input_dim

            Corr_matrix = random_correlation.rvs(eig_2, random_state=1)

            self.sigma_vector = np.random.RandomState(2).uniform(min_vol, max_vol,self.input_dim).reshape(-1,1) 
            self.Cov_matrix = np.matmul(self.sigma_vector, self.sigma_vector.T)*Corr_matrix

            self.S0 = np.random.RandomState(0).uniform(self.lb, self.ub, self.input_dim)
        except:
            pass
        
    def sample_grid(self, Nperdim):
        grid = [
            np.linspace(self.lb, self.ub, Nperdim, dtype=np.float32)
            for _ in range(self.input_dim)
        ]

        X = np.concatenate([g.reshape(-1, 1) for g in np.meshgrid(*grid)],
                           axis=1)
        X = torch.tensor(X, dtype=torch.float32)
        Y = self.pay_func(X)
        return X, Y

    def sample_uniform(self, N, seed=None):
        X = np.random.RandomState(seed).uniform(self.lb, self.ub, (N, self.input_dim))
        X = torch.tensor(X, dtype=torch.float32)
        Y = self.pay_func(X)
        return X, Y

    def sample_lognormal(self,N, seed = None):
        diff = np.random.RandomState(seed).multivariate_normal(
                                np.zeros(self.input_dim), self.Cov_matrix, size = N)
        X = self.S0.reshape(1,-1)*np.exp((self.r-self.sigma_vector.reshape(1,-1)**2/2)*self.T + self.sigma_vector.reshape(1,-1)*np.sqrt(self.T)*diff)
        X = torch.tensor(X, dtype=torch.float32)
        Y = self.pay_func(X)
        return X, Y