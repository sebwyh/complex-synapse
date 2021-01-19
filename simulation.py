import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random
from neuron import Log
import scipy.stats as stats


class Simulator:
    def __init__(self, sigma_s, epsilon, sigma_y, T_e, mode='block', dt=0.01):
        self.dt = dt
        self.sigma_s = sigma_s
        self.epsilon = epsilon
        self.sigma_y = sigma_y
        self.mode = mode
        self.T_e = T_e

    def run(self, neuron, T, avg=False, summary=False, cutoff=None):
        '''

        '''

        log = Log(neuron, self.sigma_s, self.epsilon, self.sigma_y, self.T_e, self.mode, T, self.dt)

        # y = random.randn(neuron.N, 1) # principle component
        # y = y / np.sqrt(y.T @ y)
        # orthog = random.randn(neuron.N, 1) 
        # orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
        # orthog = orthog / np.sqrt(orthog.T @ orthog)
        
        z = np.zeros((neuron.N, 1))
        z[0,0] = 1

        y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
        # y = z + self.sigma_y/np.sqrt(neuron.N-1) * np.reshape(random.multivariate_normal(np.zeros(neuron.N), np.identity(neuron.N)- z @ z.T), (neuron.N, 1)) # principle component
        y = y / np.sqrt(y.T @ y)
        orthog = random.randn(neuron.N, 1) 
        orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
        orthog = orthog / np.sqrt(orthog.T @ orthog)

        Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))

        neuron.W = neuron.W0
        neuron.w = neuron.W.T @ neuron.e1


        for i in range(len(log.timeline)):
            
            # if self.mode == 'block':
            #     if log.timeline[i] % self.T_e == 0:
            #         y = random.randn(neuron.N, 1) # principal component
            #         y = y / np.sqrt(y.T.dot(y))
            #         orthog = random.randn(neuron.N, 1) 
            #         orthog = orthog - ((orthog.T @ y).item()/(neuron.e1.T @ neuron.e1).item()) * y 
            #         orthog = orthog / np.sqrt(orthog.T @ orthog)

            if self.mode == 'block':
                if log.timeline[i] % self.T_e == 0:
                    y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
                    # y = z + self.sigma_y/np.sqrt(neuron.N-1) * np.reshape(random.multivariate_normal(np.zeros(neuron.N), np.identity(neuron.N)- z @ z.T), (neuron.N, 1)) # principle component
                    y = y / np.sqrt(y.T @ y)
                    orthog = random.randn(neuron.N, 1) 
                    orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
                    orthog = orthog / np.sqrt(orthog.T @ orthog)

                    Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))

            s = self.sigma_s * random.randn()
            xi = self.epsilon * random.randn(neuron.N, 1)
            u = s * y + xi
            v = u.T @ neuron.w

            if avg == False:
                Cu = u @ (u.T)
            
            neuron.W += self.dt * (1/neuron.tau_W) * (
                neuron.L @ neuron.W + neuron.P @ neuron.W @ Cu - neuron.alpha * np.trace(neuron.P @ neuron.W @ Cu @ neuron.W.T
                ) * neuron.P @ neuron.W)
            neuron.w = neuron.W.T @ neuron.e1

            log.s[i] = s
            log.v[i] = v
            log.u[i] = u
            log.y[i] = y
            log.z[i] = z
            log.orthog[i] = orthog
            log.W[i] = neuron.W
            # log.w[i] = neuron.w
            # log.w_norm[i] = np.sqrt(neuron.w.T @ neuron.w).item()
            # log.w_para[i] = (neuron.w.T @ y/np.sqrt(y.T @ y)).item()
            # log.w_orthog[i] = (neuron.w.T @ orthog/np.sqrt(orthog.T @ orthog)).item()

        
        if summary:
            cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
            d = np.minimum(1 - cos, 1 + cos)
            E = np.mean(d[int(cutoff*len(d)):])
            R = stats.pearsonr(log.v[int(cutoff*len(log.v)):] * np.sign(cos[int(cutoff * len(log.s)):]), log.s[int(cutoff * len(log.s)):])[0]
            neuron.logs.append((log.env_parameters, R, E))

        else:
            neuron.logs.append(log)
        
        neuron.reinitialise()

        return