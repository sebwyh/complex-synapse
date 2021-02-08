import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random
from neuron import Log
import scipy.stats as stats


class Simulator:
    def __init__(self, sigma_s, epsilon, sigma_y, T_e, tau_u=None, mode='block', dt=0.01):
        self.dt = dt
        self.sigma_s = sigma_s
        self.epsilon = epsilon
        self.sigma_y = sigma_y
        self.mode = mode
        self.T_e = T_e
        self.tau_u = tau_u
        if tau_u: self.a = dt/tau_u
        self.b = dt/T_e

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


        # print((self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N)- z @ z.T))
        # print(np.linalg.cholesky((self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N)- z @ z.T) + 1e-8 * np.eye(neuron.N)) @ np.random.randn(neuron.N, 1))
        y1 = np.linalg.cholesky((self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N)- z @ z.T) + 1e-8 * np.eye(neuron.N)) @ np.random.randn(neuron.N, 1)
        # y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
        # y = z + self.sigma_y/np.sqrt(neuron.N-1) * np.reshape(random.multivariate_normal(np.zeros(neuron.N), np.identity(neuron.N)- z @ z.T), (neuron.N, 1)) # principle component
        
        y = z + y1
        y = y / np.sqrt(y.T @ y)
        orthog = random.randn(neuron.N, 1) 
        orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
        orthog = orthog / np.sqrt(orthog.T @ orthog)
        # print(y)
        # print(y.T @ y)
        # print(y.T @ z)

        Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))

        s = self.sigma_s * random.randn()
        xi = self.epsilon * random.randn(neuron.N, 1)
        u = s * y + xi        

        neuron.W = neuron.W0
        neuron.w = neuron.W.T @ neuron.e1

        next_u = 0

        chol_K = np.sqrt(self.b) * np.linalg.cholesky(2  * (self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N)- z @ z.T) + 1e-8 * np.eye(neuron.N))
        # chol_K = np.linalg.cholesky(2 * self.b * (self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N))) 
        # print(chol_K)

        for i in range(len(log.timeline)):
            
        
            '''complete random walk'''

            if self.mode == 'block':
                if log.timeline[i] % self.T_e == 0:
                    # y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
                    y = z + np.linalg.cholesky((self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N)- z @ z.T) + 1e-8 * np.eye(neuron.N)) @ np.random.randn(neuron.N, 1)
                    y = y / np.sqrt(y.T @ y)
                    orthog = random.randn(neuron.N, 1) 
                    orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
                    orthog = orthog / np.sqrt(orthog.T @ orthog)

                    Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))

                if not self.tau_u:
                    s = self.sigma_s * random.randn()
                    xi = self.epsilon * random.randn(neuron.N, 1)
                    u = s * y + xi

                elif log.timeline[i] == next_u:
                    next_u += self.tau_u
                    s = self.sigma_s * random.randn()
                    xi = self.epsilon * random.randn(neuron.N, 1)
                    u = s * y + xi
 
            if self.mode == 'ou':
                rand_y1 = chol_K @ np.random.randn(neuron.N, 1)
                # rand_y = np.sqrt(self.dt) * z + chol_K @ np.random.randn(neuron.N, 1)

                # rand_y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1))
                # rand_y = rand_y / np.sqrt(rand_y.T @ rand_y)

                # print(rand_y)
                # y = (1-self.b) * y + self.b * rand_y
                y1 = (1-self.b) * y1 + rand_y1
                y = z + y1
                y = y / np.sqrt(y.T @ y)
                # print(y.T @ z / np.sqrt(y.T @ y))
                # print(np.sqrt(y.T @ y))

                s = self.sigma_s * random.randn()
                xi = self.epsilon * random.randn(neuron.N, 1)
                rand_u = np.sqrt(self.a * 2) * (s * y + xi)
                # u = (1-self.a) * u + self.a * rand_u
                u = (1-self.a) * u + rand_u
                # print(np.sqrt(u.T @ u))
                # print(y.T @ z / np.sqrt(y.T @ y), u.T @ z / np.sqrt(u.T @ u))

            '''smoothed block change'''
            # if log.timeline[i] % self.T_e == 0:
            #     rand_y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
            #     # y = z + self.sigma_y/np.sqrt(neuron.N-1) * np.reshape(random.multivariate_normal(np.zeros(neuron.N), np.identity(neuron.N)- z @ z.T), (neuron.N, 1)) # principle component
            #     rand_y = rand_y / np.sqrt(rand_y.T @ rand_y)
            #     if self.mode == 'block':
            #         y = rand_y
            #         orthog = random.randn(neuron.N, 1) 
            #         orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
            #         orthog = orthog / np.sqrt(orthog.T @ orthog)
            #         Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))

            # if not self.tau_u:
            #     s = self.sigma_s * random.randn()
            #     xi = self.epsilon * random.randn(neuron.N, 1)
            #     u = s * y + xi
            # elif log.timeline[i] == next_u:
            #     next_u += self.tau_u
            #     s = self.sigma_s * random.randn()
            #     xi = self.epsilon * random.randn(neuron.N, 1)
            #     rand_u = s * y + xi           
            #     if self.mode == 'block':
            #         u = rand_u

            # if self.mode == 'ou':
            #     # rand_y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1))
            #     # rand_y = rand_y / np.sqrt(rand_y.T @ rand_y)
            #     y = (1-self.b) * y + self.b * rand_y
            #     Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))


            #     # s = self.sigma_s * random.randn()
            #     # xi = self.epsilon * random.randn(neuron.N, 1)
            #     # rand_u = s * y + xi      
                
                
            #     u = (1-self.a) * u + self.a * rand_u


            
            
            
            
            v = u.T @ neuron.w
            # print(np.sqrt(v.T @ v))

            if avg == False:
                Cu = u @ (u.T)
            
            neuron.W += self.dt * (1/neuron.tau_W) * (
                neuron.L @ neuron.W + neuron.P @ neuron.W @ Cu - neuron.alpha * np.trace(neuron.P @ neuron.W @ Cu @ neuron.W.T
                ) * neuron.P @ neuron.W)
            neuron.w = neuron.W.T @ neuron.e1

            # print(neuron.w.T)
            # print(neuron.L @ neuron.W)
            # print(neuron.P @ neuron.W @ Cu)
            # print(neuron.P @ neuron.W @ Cu @ neuron.W.T)
            # print('\n')

            # if np.isnan(neuron.w).any():

            #     return

            

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
            # if np.isnan(log.v).any() or np.isinf(log.v).any():
            #     print('v')
            #     print(log.v)
            # if np.isnan(cos).any() or np.isinf(cos).any():
            #     print('cos')
            #     print(cos)
            #     print(np.transpose(log.W[:, [0], :], (0,2,1)))
            # if np.isnan(log.s).any() or np.isinf(log.s).any():
            #     print('s')
            #     print(log.s)
            R = stats.pearsonr(log.v[int(cutoff*len(log.v)):] * np.sign(cos[int(cutoff * len(log.s)):]), log.s[int(cutoff * len(log.s)):])[0]
            neuron.logs.append((log.env_parameters, R, E))

        else:
            neuron.logs.append(log)
        
        neuron.reinitialise()

        return