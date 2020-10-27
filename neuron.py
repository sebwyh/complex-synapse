import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random


class Log:
    def __init__(self, sigma, epsilon, T_e, mode, T, dt):
        self.env_parameters = dict(
            sigma = sigma,
            epsilon = epsilon,
            T_e = T_e,
            mode = mode,
            T = T,
            dt = dt
        )
        self.timeline = []
        self.s = []
        self.u = []
        self.v = []
        self.y = []
        self.orthog = []
        self.W = []
        self.w = []
        self.w_norm = []
        self.w_para = []
        self.w_orthog = []


    def __repr__(self):
        return 'sigma={}, epsilon={}'.format(
            self.env_parameters['sigma'], 
            self.env_parameters['epsilon']
        )


class Neuron:
    def __init__(self, N, S, tau_W, beta, n=2, alpha=1):
        self.hyper = dict(
            N = N,
            S = S,
            tau_W = tau_W,
            beta = beta,
            n = n,
            alpha = alpha
        )
        self.e1 = np.zeros((S, 1))
        self.e1[0,0] = 1
        self.P = self.e1 @ (self.e1.T)
        self.N = N
        self.S = S
        self.tau_W = tau_W
        self.alpha = alpha
        self.beta = beta
        self.L = np.zeros((S, S))
        for a in range(S):
            self.L[a, a] = -beta * (n ** (-2*(a+1) + 1) * (n + 1))
            if a == 0:
                self.L[a, a] = -beta * (n ** (-2*(a+1) + 1))
            if a != 0:
                self.L[a, a-1] = beta * (n ** (-2*(a+1) + 2))
            if a != S-1:
                self.L[a, a+1] = beta * (n ** (-2*(a+1) + 1))
        self.W0 = random.randn(S, N)
        self.W = self.W0
        self.w = self.W.T @ self.e1
        self.logs = []

    def __repr__(self):
        output = 'Properties: {}.\nTrials: \n'.format(self.hyper)
        if len(self.logs) != 0:
            for i in range(len(self.logs)):
                output += '{}: {}. \n'.format(i, self.logs[i].env_parameters)
        return output
        
    
    def plot(self):
        pass

    def reinitialise(self):
        self.W0 = random.randn(self.S, self.N)
        return


