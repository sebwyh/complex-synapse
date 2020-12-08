import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random


class Log:
    def __init__(self, neuron, sigma_s, epsilon, sigma_y, T_e, mode, T, dt):
        self.env_parameters = dict(
            sigma_s = sigma_s,
            epsilon = epsilon,
            sigma_y = sigma_y,
            T_e = T_e,
            mode = mode,
            T = T,
            dt = dt
        )
        self.timeline = np.arange(0, T, dt)
        length = len(self.timeline)
        self.s = np.zeros(length)
        self.u = np.zeros((length, neuron.N, 1))
        self.v = np.zeros(length)
        self.y = np.zeros((length, neuron.N, 1))
        self.z = np.zeros((length, neuron.N, 1))
        self.orthog = np.zeros((length, neuron.N, 1))
        self.W = np.zeros((length, neuron.S, neuron.N))
        self.w = np.zeros((length, neuron.N, 1))
        self.w_norm = np.zeros(length)
        self.w_para = np.zeros(length)
        self.w_orthog = np.zeros(length)


    def __repr__(self):
        return '{}'.format(
            self.env_parameters
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
        self.W0 = random.randn(S, N) / np.sqrt(N) * 0.01
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
        self.W0 = random.randn(self.S, self.N)  / np.sqrt(self.N) * 0.01
        return

    @staticmethod
    def make_neurons():
        return

