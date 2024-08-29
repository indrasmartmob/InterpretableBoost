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
    pair_thres_ = {tuple(vars): set() for vars in pair_vars_index}
    tree_vars_set = {tuple(vars):set() for vars in pair_vars_index}
    single_node_intercept = 0

    try:
        for i_tree, tree_info in enumerate(model_info):
            if "leaf" in tree_info:
                single_nodel_intercept = single_node_intercept + tree_info["leaf"]
            else:
                next_sub_tree = []
                tree_info["split_vars"] = set()
                tree_info["split_vals"] = []
                tree_info["split_path"] = []
                tree_info["missing_path"] = []
                sub_tree_info = [tree_info]
                while sub_tree_info:
                    for sub_tree in sub_tree_info:
                        split_vars = sub_tree["split_vars"].copy()
                        if "leaf" in sub_tree:
                            if len(split_vars) == 1:
                                if sub_tree["first_child"] == "yes":
                                    single_effect_index.add(tuple(split_vars))
                                    single_thres[(sub_tree["split_path"][0], sub_tree["split_path"][0])].add(tuple([val for _, val in sorted(zip(sub_tree["split_path"], sub_tree["split_vals"]))]))
                                    tree_var_set[tuple(split_vars)[0]].add(i_tree)
                            else:
                                if sub_tree["first_child"] == "yes":
                                    pair_effect_index.add(tuple(sorted(split_vars)))
                                    pair_thres_[tuple(sorted(sub_tree["split_vars"]))].update(sorted(zip(sub_tree["split_path"], sub_tree["split_vals"])))
                                    tree_vars_set[tuple(sorted(split_vars))].add(i_tree)
                        else:
                            first_child_ind = False
                            for i_child, child in enumerate(sub_tree["children"]):
                                if (first_child_ind is False) & ("leaf" in child):
                                    child["first_child"] = "yes"
                                    first_child_ind = True
                                else:
                                    child["first_child"] = "no"
                                split_var = sub_tree["split"]
                                child["missing_path"] = sub_tree["missing_path"].copy()
                                if child["nodeid"] == sub_tree["missing"]:
                                    child["missing_path"].append(True)
                                else:
                                    child["missing_path"].append(False)
                                
                                child["split_path"] = sub_tree["split_path"] + [split_var]
                                split_vals = sub_tree["split_vals"].copy()
                                split_vals.append(sub_tree["split_condition"])
                                if split_var not in split_vars:
                                    split_vars.add(split_var)
                                child["split_vars"] = split_vars
                                child["split_vals"] = split_vals
                                next_sub_tree.append(child)
                    sub_tree_info = next_sub_tree
                    next_sub_tree = []
    except:
        print("Please load an iXGB2 model with pairwise interactions only")
        return
    
    pair_thres = {}
    for key, value in pair_thres_.items():
        dic = dict()
        for k, v in value:
            if k in dic:
                dic[k].append(v)
            else:
                dic[k] = [v]
        pair_thres[key] = dic

    single_effect_index = [index[0] for index in single_effect_index]
    pair_effect_index = list(pair_effect_index)

    single_thres = {var: single_thres[tuple((var, var))] for var in single_effect_index}
    pair_thres = {tuple(vars): pair_thres[tuple(sorted(vars))] for vars in pair_effect_index}

    ### get lookup tables
    ### generate raw single lookup tables
    df_single_lookup_set = dict()
    for var in single_effect_index:
        thres_set = [np.nan]
        for thres in single_thres[var]:
            temp = [val for val in thres]
            thres_set += temp
        thres_list = sorted(set(thres_set.copy()))
        thres_array = np.array(thres_list)
        thres_list.append(np.inf)
        eval_set = np.hstack([thres_array[0], thres_array[1] - 0.01, (thres_array[1:-1] + thres_array[2:]) / 2, thres_array[-1] + 0.01])
        X_eval = np.zeros(len(eval_set) * num_cols).reshape([len(eval_set), num_cols])
        X_eval[:, model_var_list.index(var)] = eval_set
        X_eval = pd.DataFrame(X_eval, columns = model_var_list)

        num_rows = X_eval.shape[0]

        single_effect = {var: np.zeros(num_rows) for var in model_var_list}
        for i_tree in tree_var_set[var]:
            tree_info = model_info[i_tree]
            next_sub_tree = []
            tree_info["split_vars"] = set()
            tree_info["indicator"] = np.ones(num_rows)
            sub_tree_info = [tree_info]
            while sub_tree_info:
                for sub_tree in sub_tree_info:
                    split_vars = sub_tree["split_vars"].copy()
                    if "leaf" in sub_tree:
                        if len(split_vars) == 1:
                            single_effect[tuple(split_vars)[0]] += sub_tree["indicator"] * sub_tree["leaf"]
                    else:
                        for child in sub_tree["children"]:
                            split_var = sub_tree["split"]
                            if child["nodeid"] == sub_tree["yes"]:
                                if child["nodeid"] == sub_tree["missing"]:
                                    temp = (X_eval[split_var] < sub_tree["split_condition"])|(X_eval[split_var].isna())
                                else:
                                    temp = (X_eval[split_var] < sub_tree["split_condition"])
                                child["indicator"] = sub_tree["indicator"] * temp
                            else:
                                if child["nodeid"] == sub_tree["missing"]:
                                    temp = (X_eval[split_var] >= sub_tree["split_condition"])|(X_eval[split_var].isna())
                                else:
                                    temp = (X_eval[split_var] >= sub_tree["split_condition"])
                                child["indicator"] = sub_tree["indicator"] * temp
                            next_sub_tree.append(child)
                sub_tree_info = next_sub_tree
                next_sub_tree = []
        df_single_effect = single_effect[var]
        df_single_effect.index = thres_list
        df_single_lookup_set[var] = df_single_effect.to_frame()
    
    ### generate raw pairwise lookup tables
    df_pair_lookup_set = dict()
    for pair in pair_effect_index:
        var1, var2 = pair
        thres_set1 = [np.nan] + pair_thres[tuple(pair)][var1]
        thres_set2 = [np.nan] + pair_thres[tuple(pair)][var2]

        thres1_list = sorted(set(thres_set1))
        thres1_array = np.array(thres1_list)
        thres2_list = sorted(set(thres_set2))
        thres2_array = np.array(thres2_list)

        eval_set1 = np.hstack([thres1_array[0], thres1_array[1] - 0.01, (thres1_array[1:-1] + thres1_array[2:]) / 2, thres1_array[-1] + 0.01])
        eval_set2 = np.hstack([thres2_array[0], thres2_array[1] - 0.01, (thres2_array[1:-1] + thres2_array[2:]) / 2, thres2_array[-1] + 0.01])

        X_eval = np.zeros(len(eval_set1) * len(eval_set2) * num_cols).reshape([len(eval_set1) * len(eval_set2), num_cols])
        X_eval = []
        for j, x1 in enumerate(eval_set1):
            for i, x2 in enumerate(eval_set2):
                X_eval_temp = np.zeros(num_cols)
                X_eval_temp[model_var_list.index(var1)] = x1
                X_eval_temp[model_var_list.index(var2)] = x2
                X_eval.append(X_eval_temp)
        
        thres1_list.append(np.inf)
        thres2_list.append(np.inf)

        X_eval = pd.DataFrame(np.array(X_eval), columns = model_var_list)
        num_rows = X_eval.shape[0]

        pair_effect = {tuple(vars): np.zeros(num_rows) for vars in pair_vars_index}
        for i_tree in tree_vars_set[tuple((var1, var2))]:
            tree_info = model_info[i_tree]
            

