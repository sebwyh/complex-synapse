import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random
from neuron import Log
import scipy.stats as stats


class Simulator:
    def __init__(self, sigma_s, epsilon, sigma_y, tau_y, sigma_z=None, tau_u=None, tau_z=None, mode='block', z='fixed', dt=0.01):
        self.dt = dt
        self.sigma_s = sigma_s
        self.epsilon = epsilon
        self.sigma_y = sigma_y
        self.mode = mode
        self.tau_y = tau_y
        self.tau_u = tau_u
        self.tau_z = tau_z
        if tau_u: self.a = dt/tau_u
        if tau_z: self.c = dt/tau_z
        self.b = dt/tau_y
        if z: self.z = z
        if sigma_z: self.sigma_z = sigma_z

    def run(self, neuron, T, avg=False, summary=False, cutoff=None):
        '''

        '''

        log = Log(neuron, self.sigma_s, self.epsilon, self.sigma_y, self.tau_u, self.tau_y, self.tau_z, self.mode, T, self.dt)

        # y = random.randn(neuron.N, 1) # principle component
        # y = y / np.sqrt(y.T @ y)
        # orthog = random.randn(neuron.N, 1) 
        # orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
        # orthog = orthog / np.sqrt(orthog.T @ orthog)
        
        z = np.zeros((neuron.N, 1))
        z[0,0] = 1

        # chol_y = np.linalg.cholesky((self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N)- z @ z.T) + 1e-8 * np.eye(neuron.N))

        y1 = (self.sigma_y)/np.sqrt(neuron.N-1) * np.random.randn(neuron.N, 1)

        y1 = y1 - (y1.T @ z) / (z.T @ z) * z
        
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
        neuron.w = neuron.W.T @ neuron.p

        next_u = 0
        next_y = 0

        u_intv = int(self.tau_u/self.dt)
        y_intv = int(self.tau_y/self.dt)
        if self.tau_z:
            z_intv = int(self.tau_z/self.dt)

        # chol_y = np.linalg.cholesky(2 * self.b * (self.sigma_y ** 2)/(neuron.N-1) * (np.eye(neuron.N))) 
        # print(chol_y)

        for i in range(len(log.timeline)):
            
            if self.z == 'ou':
                rand_z = (self.sigma_z)/np.sqrt(neuron.N) * np.random.randn(neuron.N, 1)
                z = (1-self.c) * z + np.sqrt(2 * self.c) * rand_z
            
            '''complete random walk'''

            if self.mode == 'block':
                if i == next_y:
                    next_y += y_intv
                    # y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
                    rand_y1 = (self.sigma_y)/np.sqrt(neuron.N-1) * np.random.randn(neuron.N, 1)
                    rand_y1 = rand_y1 - (rand_y1.T @ z) / (z.T @ z) * z
                    y = z + rand_y1
                    y = y / np.sqrt(y.T @ y)
                    orthog = random.randn(neuron.N, 1)
                    orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
                    orthog = orthog / np.sqrt(orthog.T @ orthog)

                    Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))

                if not self.tau_u:
                    s = self.sigma_s * random.randn()
                    xi = self.epsilon * random.randn(neuron.N, 1)
                    u = s * y + xi

                elif i == next_u:
                    next_u += u_intv
                    s = self.sigma_s * random.randn()
                    xi = self.epsilon * random.randn(neuron.N, 1)
                    u = s * y + xi
 
            if self.mode == 'ou':
                # rand_y1 = chol_y @ np.random.randn(neuron.N, 1)
                rand_y1 = (self.sigma_y)/np.sqrt(neuron.N-1) * np.random.randn(neuron.N, 1)
                rand_y1 = rand_y1 - (rand_y1.T @ z) / (z.T @ z) * z

                y1 = (1-self.b) * y1 + np.sqrt(2 * self.b) * rand_y1
                y = z + y1
                y = y / np.sqrt(y.T @ y)

                rand_s = self.sigma_s * random.randn()
                rand_xi = self.epsilon * random.randn(neuron.N, 1)
                s = (1-self.a) * s + np.sqrt(self.a * 2) * rand_s
                xi = (1-self.a) * xi + np.sqrt(self.a * 2) * rand_xi
                u = s * y + xi
                

            if self.mode == 'ou_u':
                if i == next_y:
                    next_y += y_intv
                    # y = np.reshape(random.multivariate_normal(np.reshape(z, (neuron.N,)), (self.sigma_y ** 2)/(neuron.N-1) * (np.identity(neuron.N)- z @ z.T)), (neuron.N, 1)) # principle component
                    rand_y1 = (self.sigma_y)/np.sqrt(neuron.N-1) * np.random.randn(neuron.N, 1)
                    rand_y1 = rand_y1 - (rand_y1.T @ z) / (z.T @ z) * z
                    y = z + rand_y1
                    y = y / np.sqrt(y.T @ y)
                    orthog = random.randn(neuron.N, 1) 
                    orthog = orthog - ((orthog.T @ y).item()/(y.T @ y).item()) * y # second principal component of covariance
                    orthog = orthog / np.sqrt(orthog.T @ orthog)

                    Cu = self.sigma_s ** 2 * (y @ y.T) + self.epsilon ** 2 * (np.eye(neuron.N))
                
                
                rand_s = self.sigma_s * random.randn()
                rand_xi = self.epsilon * random.randn(neuron.N, 1)
                s = (1-self.a) * s + np.sqrt(self.a * 2) * rand_s
                xi = (1-self.a) * xi + np.sqrt(self.a * 2) * rand_xi
                u = s * y + xi               

                
                

            '''smoothed block change'''
            # if log.timeline[i] % self.tau_y == 0:
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

            dW = neuron.L @ neuron.W
            if avg == False:
                dW[[0],:] += v * u.T - neuron.alpha * v**2 * neuron.w.T
            else:
                dW[[0], :] += neuron.w.T @ Cu - neuron.alpha * (neuron.w.T @ Cu @ neuron.w) * neuron.w.T

            neuron.W += self.dt * (1/neuron.tau_W) * dW
            neuron.w = neuron.W.T @ neuron.p

            # if avg == False:
            #     Cu = u @ (u.T)
            
            # neuron.W += self.dt * (1/neuron.tau_W) * (
            #     neuron.L @ neuron.W + neuron.P @ neuron.W @ Cu - neuron.alpha * np.trace(neuron.P @ neuron.W @ Cu @ neuron.W.T
            #     ) * neuron.P @ neuron.W)
            # neuron.w = neuron.W.T @ neuron.e1

            
            
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
            E_naive = np.mean(1-(np.transpose(log.z, (0,2,1)) @ log.y).squeeze())
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
            neuron.logs.append((log.env_parameters, R, E, E_naive))

        else:
            neuron.logs.append(log)
        
        neuron.reinitialise()

        return