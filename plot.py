import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats as stats

plt.rcParams.update({'font.size': 12})


def neuron_extract(neurons):
    N = []
    S = []
    tau_W = []
    beta = []
    n = []
    alpha = []

    for neuron in neurons:
        if neuron.hyper['N'] not in N:
            N.append(neuron.hyper['N'])
        if neuron.hyper['S'] not in S:
            S.append(neuron.hyper['S']) 
        if neuron.hyper['tau_W'] not in tau_W:
            tau_W.append(neuron.hyper['tau_W']) 
        if neuron.hyper['beta'] not in beta:
            beta.append(neuron.hyper['beta']) 
        if neuron.hyper['n'] not in n:
            n.append(neuron.hyper['n']) 
        if neuron.hyper['alpha'] not in alpha:
            alpha.append(neuron.hyper['alpha']) 
    
    return N, S, tau_W, beta, n, alpha

def env_extract(neuron):
    sigma_s = []
    epsilon = []
    sigma_y = []
    T_e = []

    for log in neuron.logs:
        if log.env_parameters['T_e'] not in T_e:
            T_e.append(log.env_parameters['T_e']) 
        if log.env_parameters['sigma_s'] not in sigma_s:
            sigma_s.append(log.env_parameters['sigma_s']) 
        if log.env_parameters['sigma_y'] not in sigma_y:
            sigma_y.append(log.env_parameters['sigma_y']) 
        if log.env_parameters['epsilon'] not in epsilon:
            epsilon.append(log.env_parameters['epsilon']) 
    
    return sigma_s, epsilon, sigma_y, T_e

def tag_R_E(ax, log, x_low, x_up):
    cutoff = -0.65
    cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
    d = np.minimum(1 - cos, 1 + cos)
    E = np.mean(d[int(cutoff*len(d)):])
    R = stats.pearsonr(log.v[int(cutoff*len(log.v)):] * np.sign(cos[int(cutoff * len(log.s)):]), log.s[int(cutoff * len(log.s)):])[0]
    ax.text(
        x_up - 0.02 * (x_up - x_low), 1, 
        '$R = {}$\n$E = {}$'.format(np.round(R, 3), np.round(E, 3)),
        ha = 'right',
        va = 'top'
    )

    


def plot_wz_align(neuron):

    sigma_s, epsilon, sigma_y, T_e = env_extract(neuron)

    cm_section = np.linspace(0.3, 1, neuron.S)
    colors = [ cm.Oranges(x) for x in cm_section ]

    fig, axs = plt.subplots(len(T_e), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(T_e)), sharey=True)

    for log in neuron.logs:
        i = T_e.index(log.env_parameters['T_e'])
        j = epsilon.index(log.env_parameters['epsilon'])
        x_low = 0
        x_up = log.timeline[-1]+1
        for k in np.arange(0, neuron.S, 1):
            axs[i, j].plot(
                log.timeline, 
                (log.W[:, [k], :] @ log.z / np.sqrt(log.W[:, [k], :] @ np.transpose(log.W[:, [k], :], (0,2,1)))).squeeze(), 
                color = colors[k],
                label = '$\mathbf{{w}}_{{{k}}}$'.format(k=k+1)
            )
            axs[i, j].set_xlabel('$t$', fontsize=14)
            axs[i, j].set_title('$T_e = {}$, $\epsilon = {}$'.format(T_e[i], epsilon[j]))
        
        tag_R_E(axs[i, j], log, x_low, x_up)
        
        axs[i, j].set_xlim(x_low, x_up)
        if j == 0:
            axs[i, j].set_ylabel('$\cos(\\theta_{{\mathbf{{z}},\mathbf{{w}}_{{a}}}})$', fontsize=14)
            if i == 0:
                axs[i, j].legend(loc="upper left")

    fig.tight_layout()
    plt.show()

    return fig

def plot_wy_align(neuron, blocks=(0,3)):

    sigma_s, epsilon, sigma_y, T_e = env_extract(neuron)

    cm_section = np.linspace(0.3, 1, neuron.S)
    colors = [ cm.Oranges(x) for x in cm_section ]

    fig, axs = plt.subplots(len(T_e), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(T_e)), sharey=True)

    for log in neuron.logs:
        i = T_e.index(log.env_parameters['T_e'])
        j = epsilon.index(log.env_parameters['epsilon'])
        x_low = blocks[0] * T_e[i]
        x_up = blocks[1] * T_e[i]
        for k in np.arange(0, neuron.S, 1):
            axs[i, j].plot(
                log.timeline, 
                (log.W[:, [k], :] @ log.y / np.sqrt(log.W[:, [k], :] @ np.transpose(log.W[:, [k], :], (0,2,1)))).squeeze(), 
                color = colors[k],
                label = '$\mathbf{{w}}_{{{k}}}$'.format(k=k+1)
            )
            axs[i, j].set_xlabel('$t$', fontsize=14)
            axs[i, j].set_title('$T_e = {}$, $\epsilon = {}$'.format(T_e[i], epsilon[j]))
        
        tag_R_E(axs[i, j], log, x_low, x_up)
        
        axs[i, j].set_xlim(x_low, x_up)
        if j == 0:
            axs[i, j].set_ylabel('$\cos(\\theta_{{\mathbf{{y}},\mathbf{{w}}_{{a}}}})$', fontsize=14)
            if i == 0:
                axs[i, j].legend(loc="upper left")

    fig.tight_layout()
    plt.show()

    return fig


def plot_simple_tau_wy(neurons, blocks=(0,3)):

    sigma_s, epsilon, sigma_y, T_e = env_extract(neurons[0])

    N, S, tau_W, beta, n, alpha = neuron_extract(neurons)

    cm_section = np.linspace(0.4, 1, len(tau_W))
    colors = [ cm.Blues(x) for x in cm_section ]

    fig, axs = plt.subplots(len(T_e), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(T_e)), sharey=True)

    for k, neuron in enumerate(neurons):

        for log in neuron.logs:
            i = T_e.index(log.env_parameters['T_e'])
            j = epsilon.index(log.env_parameters['epsilon'])
            x_low = blocks[0] * T_e[i]
            x_up = blocks[1] * T_e[i]

            axs[i, j].plot(
                log.timeline, 
                (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze(), 
                color = colors[k],
                label = '$\\tau_{{W}} = {}$'.format(neuron.hyper['tau_W'])
            )

            if k == 0:
                axs[i, j].set_xlabel('$t$', fontsize=14)
                axs[i, j].set_title('$T_e = {}$, $\epsilon = {}$'.format(T_e[i], epsilon[j]))
                
                axs[i, j].set_xlim(x_low, x_up)

            if j == 0:
                axs[i, j].set_ylabel('$\cos(\\theta_{{\mathbf{{y}},\mathbf{{w}}}})$', fontsize=14)
                if i == 0:
                    axs[i, j].legend(loc="upper left")

    fig.tight_layout()
    plt.show()

    return fig

def plot_simple_tau_wz(neurons):

    sigma_s, epsilon, sigma_y, T_e = env_extract(neurons[0])

    N, S, tau_W, beta, n, alpha = neuron_extract(neurons)

    cm_section = np.linspace(0.4, 1, len(tau_W))
    colors = [ cm.Blues(x) for x in cm_section ]

    fig, axs = plt.subplots(len(T_e), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(T_e)), sharey=True)

    for k, neuron in enumerate(neurons):

        for log in neuron.logs:
            i = T_e.index(log.env_parameters['T_e'])
            j = epsilon.index(log.env_parameters['epsilon'])
            x_low = 0
            x_up = log.timeline[-1]+1

            axs[i, j].plot(
                log.timeline, 
                (log.W[:, [0], :] @ log.z / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze(), 
                color = colors[k],
                label = '$\\tau_{{W}} = {}$'.format(neuron.hyper['tau_W'])
            )

            if k == 0:
                axs[i, j].set_xlabel('$t$', fontsize=14)
                axs[i, j].set_title('$T_e = {}$, $\epsilon = {}$'.format(T_e[i], epsilon[j]))
                
                axs[i, j].set_xlim(x_low, x_up)

            if j == 0:
                axs[i, j].set_ylabel('$\cos(\\theta_{{\mathbf{{z}},\mathbf{{w}}}})$', fontsize=14)
                if i == 0:
                    axs[i, j].legend(loc="upper left")

    fig.tight_layout()
    plt.show()

    return fig




# def plot_R(neuron, parameter, cutoff=None, **fixed_params):
#     fig, axs = plt.subplots(squeeze=True, figsize=(4,3))

#     R = []
#     parameters = []

#     for log in neuron.logs:

#         if type(log) == tuple:
#             for k, v in zip(fixed_params.keys(), fixed_params.values()):

#                 if log[0][k] != v:
#                     break

#             else:
#                 parameters.append(log[0][parameter])
#                 R.append(log[1])

#         else:
#             for k, v in zip(fixed_params.keys(), fixed_params.values()):

#                 if log.env_parameters[k] != v:
#                     break

#             else:
#                 parameters.append(log.env_parameters[parameter])
#                 cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
#                 R.append(stats.pearsonr(log.v[int(cutoff*len(log.v)):] * np.sign(cos[int(cutoff * len(log.s)):]), log.s[int(cutoff * len(log.s)):])[0])

def plot_R(fig, axs, neuron, parameter, cutoff=None, color=None, label=None, **fixed_params):

    R = []
    parameters = []

    for log in neuron.logs:

        if type(log) == tuple:
            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log[0][k] != v:
                    break

            else:
                parameters.append(log[0][parameter])
                R.append(log[1])

        else:
            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log.env_parameters[k] != v:
                    break

            else:
                parameters.append(log.env_parameters[parameter])
                cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                R.append(stats.pearsonr(log.v[int(cutoff*len(log.v)):] * np.sign(cos[int(cutoff * len(log.s)):]), log.s[int(cutoff * len(log.s)):])[0])



    # print(parameters)
    
    axs.plot(np.sort(np.array(parameters)), np.array(R)[np.argsort(np.array(parameters))], color=color, label=label)

    fig.tight_layout()
    # plt.show()

    return

def plot_E(fig, axs, neuron, parameter, cutoff=None, color=None, label=None, **fixed_params):

    E = []
    parameters = []

    for log in neuron.logs:

        if type(log) == tuple:
            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log[0][k] != v:
                    break

            else:
                parameters.append(log[0][parameter])
                E.append(log[2])

        else:
            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log.env_parameters[k] != v:
                    break

            else:
                parameters.append(log.env_parameters[parameter])
                cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                d = np.minimum(1 - cos, 1 + cos)
                E.append(np.mean(d[int(cutoff*len(d)):]))



    # print(parameters)

    
    axs.plot(np.sort(np.array(parameters)), np.array(E)[np.argsort(np.array(parameters))], color=color, label=label)

    fig.tight_layout()
    # plt.show()

    return