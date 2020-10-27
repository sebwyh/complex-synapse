import itertools
from collections import namedtuple
from collections import OrderedDict


def get_combinations(params):
    Combination = namedtuple('Combination', params.keys())
    
    combinations = []
    for v in itertools.product(*params.values()):
        combinations.append(Combination(*v))

    return combinations