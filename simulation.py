import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random
from neuron import Log


class Simulator:
    def __init__(self, sigma, epsilon, T_e, mode='block', dt=0.01):
        self.dt = dt
        self.sigma = sigma
        self.epsilon = epsilon
        self.mode = mode
        self.T_e = T_e

    def run(self, neuron, T):
        '''

        '''

        log = Log(neuron, self.sigma, self.epsilon, self.T_e, self.mode, T, self.dt)

        y = random.randn(neuron.N, 1) # principle component
        y = y / np.sqrt(y.T @ y)
        orthog = random.randn(neuron.N, 1) 
        orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
        orthog = orthog / np.sqrt(orthog.T @ orthog)

        neuron.W = neuron.W0
        neuron.w = neuron.W.T @ neuron.e1


        for i in range(len(log.timeline)):
            
            if self.mode == 'block':
                if log.timeline[i] % self.T_e == 0:
                    y = random.randn(neuron.N, 1) # principal component
                    y = y / np.sqrt(y.T.dot(y))
                    orthog = random.randn(neuron.N, 1) 
                    orthog = orthog - ((orthog.T @ y).item()/(neuron.e1.T @ neuron.e1).item()) * y 
                    orthog = orthog / np.sqrt(orthog.T @ orthog)

            s = self.sigma * random.randn()
            xi = self.epsilon * random.randn(neuron.N, 1)
            u = s * y + xi
            v = u.T @ neuron.w
            C = u @ (u.T)
            neuron.W += self.dt * (1/neuron.tau_W) * (
                neuron.L @ neuron.W + neuron.P @ neuron.W @ C - neuron.alpha * np.trace(neuron.P @ neuron.W @ C @ neuron.W.T
                ) * neuron.P @ neuron.W)
            neuron.w = neuron.W.T @ neuron.e1

            log.s[i] = s
            log.v[i] = v
            log.u[i] = u
            log.y[i] = y
            log.orthog[i] = orthog
            log.W[i] = neuron.W
            log.w[i] = neuron.w
            log.w_norm[i] = np.sqrt(neuron.w.T @ neuron.w).item()
            log.w_para[i] = (neuron.w.T @ y/np.sqrt(y.T @ y)).item()
            log.w_orthog[i] = (neuron.w.T @ orthog/np.sqrt(orthog.T @ orthog)).item()

        #     log.timeline.append(t)
        #     log.s.append(s)
        #     log.v.append(v)
        #     log.u.append(u)
        #     log.y.append(y)
        #     log.orthog.append(orthog)
        #     log.W.append(neuron.W)
        #     log.w.append(neuron.w)
        #     log.w_norm.append(np.sqrt(neuron.w.T @ neuron.w))
        #     log.w_para.append(neuron.w.T @ y/np.sqrt(y.T @ y))
        #     log.w_orthog.append(neuron.w.T @ orthog/np.sqrt(orthog.T @ orthog))

        # log.s = np.array(log.s).squeeze()
        # log.v = np.array(log.v).squeeze()
        # log.u = np.array(log.u)
        # log.y = np.array(log.y)
        # log.orthog = np.array(log.orthog)
        # log.W = np.array(log.W)
        # log.w = np.array(log.w)
        # log.w_norm = np.array(log.w_norm).squeeze()
        # log.w_para = np.array(log.w_para).squeeze()
        # log.w_orthog = np.array(log.w_orthog).squeeze()

        
        neuron.logs.append(log)

        return