import itertools
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import matplotlib.colors as c


def get_combinations(params):
    Combination = namedtuple('Combination', params.keys())
    
    combinations = []
    for v in itertools.product(*params.values()):
        combinations.append(Combination(*v))

    return combinations

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = c.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap