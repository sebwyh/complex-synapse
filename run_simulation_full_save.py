import numpy as np 
import numpy.random as random
import scipy
from neuron import Neuron, Log
from simulation import Simulator
from util import get_combinations
from collections import namedtuple
from collections import OrderedDict
from plot import *
import pickle
from tqdm import tqdm
from matplotlib import cm

def main():

    
    '''initialise complex model(s)'''
    neuron_parameters = OrderedDict(
        N = [10],
        S = [3],
        tau_W=[1],
        beta = [1],
        n_C = [2],
        n_g = [2],
        gamma = [1],
        phi = [10, 2, 0.5, 0.1, -0.1, -0.5, -2]
        # phi= [10]
    )

    complex_neurons = []
    for p in get_combinations(neuron_parameters):
        complex_neurons.append(Neuron(p.N, p.S, p.tau_W, p.beta, gamma=p.gamma, phi=p.phi, n_C=p.n_C, n_g=p.n_g))


    '''initialise simple models'''
    # simple_neuron_parameters = OrderedDict(
    #     N = [10],
    #     S = [1],
    #     tau_W=[0.63, 3.27, 14.54, 61.17, 295.59],
    #     # tau_W= [  0.43580738,   1.97280049,   8.02023061,  32.1402713 , 152.62443861],
    #     # tau_W=[0.63],
    #     # tau_W = [ 0.65486921,  2.31590864,  6.78519102, 19.00408772, 56.05794404],
    #     beta = [0],
    #     n_C = [2]
    # )

    # simple_neurons_for_summary = []
    # for p in get_combinations(simple_neuron_parameters):
    #     simple_neurons_for_summary.append(Neuron(p.N, p.S, p.tau_W, p.beta, p.n_C))



    '''define environment: O-U for both u and y'''
    env_parameters_ou = OrderedDict(
        # sigma_s = [0.1],
        sigma_s = [1],
        # epsilon = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2],
        epsilon = [0.1, 2],
        sigma_y = [0.5],
        # sigma_y = [0],
        # tau_y = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20],
        tau_y = [0.1, 20],
        tau_u = [0.1]
    )

    envs_for_summary_ou = []
    for p in get_combinations(env_parameters_ou):
        envs_for_summary_ou.append(Simulator(p.sigma_s, p.epsilon, p.sigma_y, p.tau_y, tau_u=p.tau_u, mode='ou', dt=0.002))

    
    
    '''run simluation with complex model(s) (full save, one file per neuron)'''
    
    for i in range(len(complex_neurons)):
    
        trials_complex_summary_ou = OrderedDict(
            neuron = [complex_neurons[i]],
            env = envs_for_summary_ou
        )

        T = 1000

        for trial in tqdm(get_combinations(trials_complex_summary_ou)):
            trial.env.run(trial.neuron, T=T, summary=False, cutoff=-0.6)

        with open('../data/complex_ou_example24-{}.pkl'.format(i), 'wb+') as f:
            pickle.dump([complex_neurons[i]], f)

        complex_neurons[i] = 0
    

    '''run simluation with simple models'''
    # trials_simple_summary_ou = OrderedDict(
    #     neuron = simple_neurons_for_summary,
    #     env = envs_for_summary_ou
    # )

    # T = 1000

    # for trial in tqdm(get_combinations(trials_simple_summary_ou)):
    #     trial.env.run(trial.neuron, T=T, summary=True, cutoff=-0.6)


    '''save'''
    # with open('../data/complex_ou21.pkl', 'wb+') as f:
    #     pickle.dump(complex_neurons, f)

    # with open('../data/complex_ou_example20.pkl', 'wb+') as f:
    #     pickle.dump(complex_neurons, f)

    # with open('../data/simple_ou20.pkl', 'wb+') as f:
    #     pickle.dump(simple_neurons_for_summary, f)


    return



if __name__== '__main__':
    main()