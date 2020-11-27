import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats as stats

plt.rcParams.update({'font.size': 12})

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
    cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
    d = np.minimum(1 - cos, 1 + cos)
    E = np.mean(d)
    R = stats.pearsonr(log.v * np.sign(cos), log.s)[0]    
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
