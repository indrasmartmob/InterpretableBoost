'''
classification + regression version
'''
import pandas as pd
import numpy as np
import time
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import ticker
from matplotlib import gridspec
from IPython.core import display as ICD
import json
from pickle import dump, load
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.stats.weightstats import DescrStatsW
from itertools import product, combinations
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def expit(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return np.log(x / (1-x))

def aggregation_algorithm(model_var_list, model_info, num_cols):

    """aggregation algorithm.

    Parameters
    --------------
    model_var_list: list
        feature list
    model_info: dictionary
        saved tree structure
    num_cols: integer
        number of columns
    """

    single_effect_index = set()
    single_thres = {tuple((var, var)): set() for var in model_var_list}
    tree_var_set = {var:set() for var in model_var_list}

    pair_vars_index = []
    for i_var1, var1 in enumerate(model_var_list):
        for var2 in model_var_list[i_var1 + 1:]:
            pair_vars_index.append(sorted(tuple((var1, var2))))
    
    pair_effect_index = set()
    