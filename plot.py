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
    tau_y = []

    for log in neuron.logs:
        if log.env_parameters['tau_y'] not in tau_y:
            tau_y.append(log.env_parameters['tau_y']) 
        if log.env_parameters['sigma_s'] not in sigma_s:
            sigma_s.append(log.env_parameters['sigma_s']) 
        if log.env_parameters['sigma_y'] not in sigma_y:
            sigma_y.append(log.env_parameters['sigma_y']) 
        if log.env_parameters['epsilon'] not in epsilon:
            epsilon.append(log.env_parameters['epsilon']) 
    
    return sigma_s, epsilon, sigma_y, tau_y

def tag_R_E(ax, log, x_low, x_up):
    cutoff = -0.6
    cos = (log.W[:, [0], :] @ log.y / (np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y)))).squeeze()
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

    sigma_s, epsilon, sigma_y, tau_y = env_extract(neuron)

    cm_section = np.linspace(0.3, 1, neuron.S)
    colors = [ cm.Oranges(x) for x in cm_section ]

    fig, axs = plt.subplots(len(tau_y), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(tau_y)), sharey=True)

    for log in neuron.logs:
        i = tau_y.index(log.env_parameters['tau_y'])
        j = epsilon.index(log.env_parameters['epsilon'])
        x_low = 0
        x_up = log.timeline[-1]+1
        for k in np.arange(0, neuron.S, 1):
            axs[i, j].plot(
                log.timeline, 
                (log.W[:, [k], :] @ log.z / (np.sqrt(log.W[:, [k], :] @ np.transpose(log.W[:, [k], :], (0,2,1))) * np.sqrt(np.transpose(log.z, (0,2,1)) @ log.z))).squeeze(), 
                # (neuron.po.T[None, :, :] @ log.W @ log.z / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.z, (0,2,1)) @ log.z))).squeeze(),
                color = colors[k],
                label = '$\mathbf{{w}}_{{{k}}}$'.format(k=k+1)
            )
            axs[i, j].set_xlabel('$t$', fontsize=14)
            axs[i, j].set_title('$\\tau_y = {}$, $\epsilon = {}$'.format(tau_y[i], epsilon[j]))
        
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

    sigma_s, epsilon, sigma_y, tau_y = env_extract(neuron)

    cm_section = np.linspace(0.3, 1, neuron.S)
    colors = [ cm.Oranges(x) for x in cm_section ]

    fig, axs = plt.subplots(len(tau_y), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(tau_y)), sharey=True)

    for log in neuron.logs:
        i = tau_y.index(log.env_parameters['tau_y'])
        j = epsilon.index(log.env_parameters['epsilon'])
        x_low = blocks[0] * tau_y[i]
        x_up = blocks[1] * tau_y[i]
        for k in np.arange(0, neuron.S, 1):
            axs[i, j].plot(
                log.timeline, 
                (log.W[:, [k], :] @ log.y / (np.sqrt(log.W[:, [k], :] @ np.transpose(log.W[:, [k], :], (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze(), 
                # (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze(),
                color = colors[k],
                label = '$\mathbf{{w}}_{{{k}}}$'.format(k=k+1)
            )
            axs[i, j].set_xlabel('$t$', fontsize=14)
            axs[i, j].set_title('$\\tau_y = {}$, $\epsilon = {}$'.format(tau_y[i], epsilon[j]))
        
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

    sigma_s, epsilon, sigma_y, tau_y = env_extract(neurons[0])

    N, S, tau_W, beta, n, alpha = neuron_extract(neurons)

    cm_section = np.linspace(0.4, 1, len(tau_W))
    colors = [ cm.Blues(x) for x in cm_section ]

    fig, axs = plt.subplots(len(tau_y), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(tau_y)), sharey=True)

    for k, neuron in enumerate(neurons):

        for log in neuron.logs:
            i = tau_y.index(log.env_parameters['tau_y'])
            j = epsilon.index(log.env_parameters['epsilon'])
            x_low = blocks[0] * tau_y[i]
            x_up = blocks[1] * tau_y[i]

            axs[i, j].plot(
                log.timeline, 
                # (log.W[:, [0], :] @ log.y / (np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze(), 
                (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze(),
                color = colors[k],
                label = '$\\tau_{{W}} = {}$'.format(neuron.hyper['tau_W'])
            )

            if k == 0:
                axs[i, j].set_xlabel('$t$', fontsize=14)
                axs[i, j].set_title('$\\tau_y = {}$, $\epsilon = {}$'.format(tau_y[i], epsilon[j]))
                
                axs[i, j].set_xlim(x_low, x_up)

            if j == 0:
                axs[i, j].set_ylabel('$\cos(\\theta_{{\mathbf{{y}},\mathbf{{w}}}})$', fontsize=14)
                if i == 0:
                    axs[i, j].legend(loc="upper left")

    fig.tight_layout()
    plt.show()

    return fig

def plot_simple_tau_wz(neurons):

    sigma_s, epsilon, sigma_y, tau_y = env_extract(neurons[0])

    N, S, tau_W, beta, n, alpha = neuron_extract(neurons)

    cm_section = np.linspace(0.4, 1, len(tau_W))
    colors = [ cm.Blues(x) for x in cm_section ]

    fig, axs = plt.subplots(len(tau_y), len(epsilon), squeeze=True, figsize=(4*len(epsilon),3*len(tau_y)), sharey=True)

    for k, neuron in enumerate(neurons):

        for log in neuron.logs:
            i = tau_y.index(log.env_parameters['tau_y'])
            j = epsilon.index(log.env_parameters['epsilon'])
            x_low = 0
            x_up = log.timeline[-1]+1

            axs[i, j].plot(
                log.timeline, 
                # (log.W[:, [0], :] @ log.z / (np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1))) * np.sqrt(np.transpose(log.z, (0,2,1)) @ log.z))).squeeze(), 
                (neuron.po.T[None, :, :] @ log.W @ log.z / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.z, (0,2,1)) @ log.z))).squeeze(),
                color = colors[k],
                label = '$\\tau_{{W}} = {}$'.format(neuron.hyper['tau_W'])
            )

            if k == 0:
                axs[i, j].set_xlabel('$t$', fontsize=14)
                axs[i, j].set_title('$\\tau_y = {}$, $\epsilon = {}$'.format(tau_y[i], epsilon[j]))
                
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
                # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                cos = (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                R.append(stats.pearsonr(log.v[int(cutoff*len(log.v)):] * np.sign(cos[int(cutoff * len(log.s)):]), log.s[int(cutoff * len(log.s)):])[0])



    # print(parameters)
    
    axs.plot(np.sort(np.array(parameters)), np.array(R)[np.argsort(np.array(parameters))], color=color, label=label)

    fig.tight_layout()
    # plt.show()

    return

def plot_E(fig, axs, neuron, parameter, cutoff=None, color=None, style='-', label=None, **fixed_params):

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
                # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                cos =  (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                d = np.minimum(1 - cos, 1 + cos)
                E.append(np.mean(d[int(cutoff*len(d)):]))

    # print(parameters)

    
    axs.plot(np.sort(np.array(parameters)), np.array(E)[np.argsort(np.array(parameters))], style, color=color, label=label)

    fig.tight_layout()
    # plt.show()

    return

def plot_cm_E_abs(fig, axs, neuron, cutoff=None, color=None, style='-', label=None, **fixed_params):

    '''different rows: different epsilon values; different columns: different tau_y values'''

    epsilons = []
    tau_ys = []


    # make two lists first, one of all epsilon values, one of all tau_y values
    for log in neuron.logs:

        if type(log) == tuple:
            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log[0][k] != v:
                    break

            else:

                if log[0]['epsilon'] not in epsilons and log[0]['epsilon'] < 1.5:

                    epsilons.append(log[0]['epsilon'])
                
                if log[0]['tau_y'] not in tau_ys and log[0]['tau_y'] > 0.25:
                    tau_ys.append(log[0]['tau_y'])

        else:

            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log.env_parameters[k] != v:
                    break

            else:

                if log.env_parameters['epsilon'] not in epsilons and log.env_parameters['epsilon'] < 1.5:

                    epsilons.append(log.env_parameters['epsilon'])
                
                if log.env_parameters['tau_y'] not in tau_ys and log.env_parameters['tau_y'] > 0.25:
                
                    tau_ys.append(log.env_parameters['tau_y'])


    # print(parameters)

    E = np.zeros((len(epsilons), len(tau_ys)))

    for log in neuron.logs:
        
        if type(log) == tuple:

            if log[0]['epsilon'] < 1.5 and log[0]['tau_y'] > 0.25:

                E[epsilons.index(log[0]['epsilon']), tau_ys.index(log[0]['tau_y'])] = log[2]

        else:

            if log.env_parameters['epsilon'] < 1.5 and log.env_parameters['tau_y'] > 0.25:

                # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                cos =  (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                d = np.minimum(1 - cos, 1 + cos)
                E[epsilons.index(log.env_parameters['epsilon']), tau_ys.index(log.env_parameters['tau_y'])] = np.mean(d[int(cutoff*len(d)):])
            
    # print(E.shape)
    # print(E)

    im = axs.imshow(E, cmap=cm.binary)
    im.set_clim(0, 1)
    axs.set_xlabel('$\\tau_y$')
    axs.set_ylabel('$\epsilon$')
    axs.set_xticks(np.arange(len(tau_ys)))
    axs.set_xticklabels(tau_ys)
    axs.set_yticks(np.arange(len(epsilons)))
    axs.set_yticklabels(epsilons)
    axs.set_title('Complex model', fontsize=12)


    fig.tight_layout()
    # plt.show()

    return im

def plot_cm_E_minus_reference(fig, axs, neuron, ref=None, cutoff=None, color=None, style='-', label=None, **fixed_params):

    '''different rows: different epsilon values; different columns: different tau_y values'''

    epsilons = []
    tau_ys = []


    # make two lists first, one of all epsilon values, one of all tau_y values
    for log in neuron.logs:


        if type(log) == tuple:

            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log[0][k] != v:
                    break

            else:

                if log[0]['epsilon'] not in epsilons and log[0]['epsilon'] < 1.5:

                    epsilons.append(log[0]['epsilon'])
                
                if log[0]['tau_y'] not in tau_ys and log[0]['tau_y'] > 0.25:

                    tau_ys.append(log[0]['tau_y'])

        else:

            ref = np.mean(1-(np.transpose(log.z, (0,2,1)) @ log.y / np.sqrt(np.transpose(log.z, (0,2,1)) @ log.z)).squeeze())

            for k, v in zip(fixed_params.keys(), fixed_params.values()):

                if log.env_parameters[k] != v:
                    break

            else:

                if log.env_parameters['epsilon'] not in epsilons and log.env_parameters['epsilon'] < 1.5:

                    epsilons.append(log.env_parameters['epsilon'])
                
                if log.env_parameters['tau_y'] not in tau_ys and log.env_parameters['tau_y'] > 0.25:
                
                    tau_ys.append(log.env_parameters['tau_y'])


    # print(parameters)

    E = np.zeros((len(epsilons), len(tau_ys)))

    for log in neuron.logs:
        
        if type(log) == tuple:

            if log[0]['epsilon'] < 1.5 and log[0]['tau_y'] > 0.25:

                if len(log) > 3:
                    
                    ref = log[3]

                E[epsilons.index(log[0]['epsilon']), tau_ys.index(log[0]['tau_y'])] = log[2] - ref

        else:

            if log.env_parameters['epsilon'] < 1.5 and log.env_parameters['tau_y'] > 0.25:

                ref = np.mean(1-(np.transpose(log.z, (0,2,1)) @ log.y).squeeze())

                # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                cos =  (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                d = np.minimum(1 - cos, 1 + cos)
                E[epsilons.index(log.env_parameters['epsilon']), tau_ys.index(log.env_parameters['tau_y'])] = np.mean(d[int(cutoff*len(d)):]) - ref
            
    # print(E.shape)
    # print(E)

    im = axs.imshow(E, cmap=cm.bwr)
    im.set_clim(-0.7, 0.7)
    axs.set_xlabel('$\\tau_y$')
    axs.set_ylabel('$\epsilon$')
    axs.set_xticks(np.arange(len(tau_ys)))
    axs.set_xticklabels(tau_ys)
    axs.set_yticks(np.arange(len(epsilons)))
    axs.set_yticklabels(epsilons)
    axs.set_title('$E-E_0$ (complex model)', fontsize=12)


    fig.tight_layout()
    # plt.show()

    return im

def plot_cm_E_diff(fig, axs, neuron1, neuron2, cutoff=None, color=None, style='-', label=None, **fixed_params):

    '''Different rows: different epsilon values; different columns: different tau_y values. neuron1 is the benchmark'''

    epsilons = []
    tau_ys = []


    # make two lists first, one of all epsilon values, one of all tau_y values
    for log in neuron1.logs:

        if type(log) == tuple:

            if log[0]['epsilon'] < 1.5 and log[0]['tau_y'] > 0.25:

                for k, v in zip(fixed_params.keys(), fixed_params.values()):

                    if log[0][k] != v:
                        break

                else:

                    if log[0]['epsilon'] not in epsilons and log[0]['epsilon'] < 1.5:

                        epsilons.append(log[0]['epsilon'])
                    
                    if log[0]['tau_y'] not in tau_ys and log[0]['tau_y'] > 0.25:
                        tau_ys.append(log[0]['tau_y'])

        else:

            if log.env_parameters['epsilon'] < 1.5 and log.env_parameters['tau_y'] > 0.25:

                for k, v in zip(fixed_params.keys(), fixed_params.values()):

                    if log.env_parameters[k] != v:
                        break

                else:

                    if log.env_parameters['epsilon'] not in epsilons and log.env_parameters['epsilon'] < 1.5:

                        epsilons.append(log.env_parameters['epsilon'])
                    
                    if log.env_parameters['tau_y'] not in tau_ys and log.env_parameters['tau_y'] > 0.25:
                    
                        tau_ys.append(log.env_parameters['tau_y'])


    E = np.zeros((len(epsilons), len(tau_ys)))

    for log in neuron2.logs:
        
        if type(log) == tuple:

            if log[0]['epsilon'] < 1.5 and log[0]['tau_y'] > 0.25:

                E[epsilons.index(log[0]['epsilon']), tau_ys.index(log[0]['tau_y'])] = log[2]

        else:

            if log.env_parameters['epsilon'] < 1.5 and log.env_parameters['tau_y'] > 0.25:

                # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                cos = (neuron2.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron2.po.T[None, :, :] @ log.W @ np.transpose(neuron2.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                d = np.minimum(1 - cos, 1 + cos)
                E[epsilons.index(log.env_parameters['epsilon']), tau_ys.index(log.env_parameters['tau_y'])] = np.mean(d[int(cutoff*len(d)):])

    for log in neuron1.logs:
        
        if type(log) == tuple:

            if log[0]['epsilon'] < 1.5 and log[0]['tau_y'] > 0.25:

                E[epsilons.index(log[0]['epsilon']), tau_ys.index(log[0]['tau_y'])] -= log[2]

        else:

            if log.env_parameters['epsilon'] < 1.5 and log.env_parameters['tau_y'] > 0.25:

                cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                cos = (neuron1.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron1.po.T[None, :, :] @ log.W @ np.transpose(neuron1.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                d = np.minimum(1 - cos, 1 + cos)
                E[epsilons.index(log.env_parameters['epsilon']), tau_ys.index(log.env_parameters['tau_y'])] -= np.mean(d[int(cutoff*len(d)):])
            
    # print(E.shape)
    # print(E)

    im = axs.imshow(E, cmap=cm.bwr)
    im.set_clim(-0.7, 0.7)
    axs.set_xlabel('$\\tau_y$')
    # axs.set_ylabel('$\epsilon$')
    axs.set_xticks(np.arange(len(tau_ys)))
    axs.set_xticklabels(tau_ys)
    # axs.set_yticks([np.arange(len(epsilons))])
    axs.set_yticks([])
    # axs.set_yticklabels(epsilons)
    axs.set_title("$\\tau_w = {}$, $\Delta\\bar{{E}}={:0.3f}$".format(neuron1.hyper['tau_W'], np.mean(E)), fontsize=12)


    fig.tight_layout()
    # plt.show()

    return im


def plot_E_average(fig, axs, neuron, parameter, cutoff=None, color=None, label=None):

    parameters = []

    for log in neuron.logs:
        
        if type(log) == tuple:

            if log[0][parameter] not in parameters:
                parameters.append(log[0][parameter])
        else:
            if log.env_parameters[parameter] not in parameters:
                parameters.append(log.env_parameters[parameter])

    parameters = np.sort(np.array(parameters))

    # print(parameters)

    E = [ [] for _ in range(len(parameters)) ]

    for log in neuron.logs:
        
        if type(log) == tuple:
            # print(np.where(parameters==log[0][parameter]))
            E[np.where(parameters==log[0][parameter])[0][0]].append(log[2])

        else:
            # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
            cos = (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
            d = np.minimum(1 - cos, 1 + cos)
        
            E[np.where(parameters==log.env_parameters[parameter])[0][0]].append(np.mean(d[int(cutoff*len(d)):]))
    
    E = np.mean(np.array(E), axis=1).squeeze()
    
    if not label: label='$\\tau_w={}$'.format(neuron.tau_W)
    axs.plot(parameters, np.array(E)[np.argsort(np.array(parameters))], color=color, label=label)

    fig.tight_layout()
    # plt.show()

    return


def plot_E_best(fig, axs, neurons, parameter, cutoff=None, color=None, label=None, **fixed_params):

    EE = []
    
    for neuron in neurons:
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
                    # cos = (log.W[:, [0], :] @ log.y / np.sqrt(log.W[:, [0], :] @ np.transpose(log.W[:, [0], :], (0,2,1)))).squeeze()
                    cos = (neuron.po.T[None, :, :] @ log.W @ log.y / (np.sqrt(neuron.po.T[None, :, :] @ log.W @ np.transpose(neuron.po.T[None, :, :] @ log.W, (0,2,1))) * np.sqrt(np.transpose(log.y, (0,2,1)) @ log.y))).squeeze()
                    d = np.minimum(1 - cos, 1 + cos)
                    E.append(np.mean(d[int(cutoff*len(d)):]))

        E = np.array(E)[np.argsort(np.array(parameters))]
        EE.append(E)

    # print(parameters)

    EE = np.array(EE)
    
    axs.plot(np.sort(np.array(parameters)), np.min(EE, axis=0), "--", color=color, label=label)

    fig.tight_layout()
    # plt.show()

    return

def plot_compare_phi_psi(neurons, epsilon_values, tau_y_values, psi_values):

    '''This only works for summary type saves.'''

    fig, axs = plt.subplots(1, len(psi_values), sharey=True, figsize=(5*len(psi_values)+2, 5))
    cm_section = np.linspace(0.3, 1, len(tau_y_values))
    colours = []
    colours.append([ cm.Blues(x) for x in cm_section ])
    colours.append([ cm.Oranges(x) for x in cm_section ])
    colours.append([ cm.Purples(x) for x in cm_section ])
    colours.append([ cm.Greens(x) for x in cm_section ])


    for j, epsilon in enumerate(epsilon_values):

        for k, tau_y in enumerate(tau_y_values):

            label = "$\\tau_y={}$, $\epsilon={}$".format(tau_y,epsilon)

            E = []
            phi_values = []

            for i in range(len(psi_values)):
                E.append([])
                phi_values.append([])

            for neuron in neurons:
                
                if neuron.hyper['psi'] in psi_values:

                    phi_values[psi_values.index(neuron.hyper['psi'])].append(neuron.hyper['phi'])

                    for log in neuron.logs:
                        
                        if log[0]['tau_y'] == tau_y and log[0]['epsilon'] == epsilon:
                    
                            E[psi_values.index(neuron.hyper['psi'])].append(log[2]-log[3])

            for i in range(len(psi_values)):
                
                if i == 0:
                    axs[i].plot(phi_values[i], E[i], label=label, color=colours[j][k])
                    axs[i].set_ylabel('$E$')
                else:
                    axs[i].plot(phi_values[i], E[i], color=colours[j][k])
                axs[i].set_title('$\psi={}$'.format(psi_values[i]))
                axs[i].set_xlabel('$\phi$')

            
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), fancybox=True, shadow=True, ncol=6)

    fig.tight_layout()
    plt.show()

    return fig