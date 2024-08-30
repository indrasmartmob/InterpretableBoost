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
            next_sub_tree = []
            tree_info["split_vars"] = set()
            tree_info["indicator"] = np.ones(num_rows)
            sub_tree_info = [tree_info]
            while sub_tree_info:
                for sub_tree in sub_tree_info:
                    split_vars = sub_tree["split_vars"].copy()
                    if "leaf" in sub_tree:
                        if len(split_vars) == 2:
                            pair_effect[tuple(sorted(split_vars))] += sub_tree["indicator"] * sub_tree["leaf"]
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
        df_pair_lookup_set[tuple((var1, var2))] = pd.DataFrame(np.array(pair_effect[tuple((var1, var2))]).reshape(len(eval_set1)), len(eval_set2).T, index = thres2_list, columns = thres1_list)

        index_set = set()
        for pair in list(tree_vars_set.keys()):
            index_set = index_set.union(pair)
        non_empty_key = [var for var in list(df_single_lookup_set) if tree_var_set[var] != set()]
        num_tree_var = {var:len(tree_var_set[var]) for var in non_empty_key}
        non_empty_keys = [pair for pair in list(df_pair_lookup_set) if tree_vars_set[pair] != set()]
        num_tree_vars = {pair:len(tree_vars_set[pair]) for pair in non_empty_keys}


    return single_effect_index, pair_effect_index, df_single_lookup_set, df_pair_lookup_set, single_node_intercept, num_tree_var, num_tree_vars

def combine_lookups(df_lu1, df_lu2):
    """Combine tow lookup tables into one.

    Parameters
    --------------
    df_lookup1: pandas dataframe
        dataframe of main effect
    df_lookup2: pandas dataframe
        dataframe of main effect
    """

    df_lookup1 = df_lu1.drop(np.nan).copy()
    df_lookup2 = df_lu2.drop(np.nan).copy()

    df_lookup_new1 = df_lookup1.copy()
    for thres in df_lookup2.index:
        if thres not in df_lookup1.index:
            val = df_lookup1.iloc[sum(df_lookup1.index < thres)]
            val.name = thres
            df_lookup_new1 = df_lookup_new1.append(val)
    df_lookup_new1 = df_lookup_new1.sort_index()

    df_lookup_new2 = df_lookup2.copy()
    for thres in df_lookup1.index:
        if thres not in df_lookup2.index:
            val = df_lookup2.iloc[sum(df_lookup2.index < thres)]
            val.name = thres
            df_lookup_new2 = df_lookup_new2.append(val)
    df_lookup_new2 = df_lookup_new2.sort_index()

    df_lookup_num = df_lookup_new1 + df_lookup_new2
    df_lookup_nan = pd.DataFrame(df_lu1.loc[np.nan, :].values + df_lu2.loc[np.nan, :].values, index = [np.nan], columns = df_lookup_num.columns)
    df_lookup_comb = pd.concat([df_lookup_nan, df_lookup_num], axis=0)

    return df_lookup_comb

def decompose_free(val_mat):
    """Data-free decomposition method"""

    intercept = val_mat.mean().mean()
    val_mat_cen = val_mat - intercept

    main_x = val_mat_cen.copy().mean(axis=0)
    main_y = val_mat_cen.copy().mean(axis=1)
    pair_xy = ((val_mat_cen - main_x).T - main_y).T

    return intercept, main_x, main_y, pair_xy

def center_row_col(dist, val_mat):
    """center rows and columns using a given joint distribution"""

    intercept = (val_mat * dist).sum().sum()
    val_mat_cen = val_mat - intercept

    cond_dist_x = dist / dist.sum(axis = 0)
    cond_dist_y = (dist.T / dist.sum(axis = 1)).T

    main_x = (val_mat_cen * cond_dist_x).sum(axis = 0)
    main_y = (val_mat_cen * cond_dist_y).sum(axis = 1)
    pair_xy = ((val_mat_cen - main_x).T - main_y).T

    max_col_row_mean = max(max(abs((pair_xy * cond_dist_y).sum(axis = 1))), max(abs((pair_xy * cond_dist_x).sum(axis = 0))))

    return intercept, main_x, main_y, pair_xy, max_col_row_mean

def decompose_ind(dist, val_mat):
    """Decomposition method with independence assumption (marginal distribution approach)"""

    dist_ind = dist.copy()
    marg_dist_x = dist.sum(axis = 0)
    marg_dist_y = dist.sum(axis = 1)
    dist_ind.iloc[:, :] = np.outer(pd.DataFrame(marg_dist_y), pd.DataFrame(marg_dist_x).T)

    return center_row_col(dist_ind, val_mat)

def decompose_full(dist, val_mat):
    """Decomposition method with full distribution"""

    max_col_row_mean = 1
    main_x_cumul = 0
    main_y_cumul = 0
    intercept_cumul = 0
    pair_xy = val_mat.copy()
    k = 0
    while ((max_col_row_mean > 1E-12) & (k < 100)):
        intercept, main_x, main_y, pair_xy, max_col_row_mean = center_row_col(dist, pair_xy)
        intercept_cumul += intercept
        main_x_cumul += main_x
        main_y_cumul += main_y
        k = k + 1
    if k == 100:
        print("No convergence")
        print(f"{k}th iteration: max col/row mean={max_col_row_mean}")
    
    return intercept_cumul, main_x_cumul, main_y_cumul, pair_xy, max_col_row_mean

def decomposition_algorithm(X, global_intercept, decomp_method, df_single_lookup_set, df_pair_lookup_set):
    """Decomposition algorithm with three decomposition methods (Distribution-free, Marginal distribution (independence assumption), and Full distribution).
    
    Parameters
    ------------
    X: pandas dataframe
        used for distribution calculation
    global_intercept: float
        global intercept
    decomp_method: string
        the decomposition method used to decompose the lookup table, chosen from 'free', 'ind', and 'full'
    df_single_lookup_set: pandas dataframe
        lookup table for main effect
    df_pair_lookup_set: pandas dataframe
        lookup table for pairwise interaction effect

    """
    if decomp_method == "free":
        #create a global intercept by adding averages from tables -- single effect
        for single in list(df_single_lookup_set.keys()):
            val_mat = df_single_lookup_set[single].copy()
            single_intercept = val_mat.mean(axis = 0).values[0]
            df_single_new = val_mat - single_intercept
            global_intercept = global_intercept + single_intercept
            df_single_lookup_set[single] = df_single_new
        
        # create a global intercept by adding averages from tables -- pairwise effect
        for pair in list(df_pair_lookup_set.keys()):
            var1, var2 = pair
            val_mat = df_pair_lookup_set[pair].copy()

            intercept_free, main_x_free, main_y_free, pair_xy_free = decompose_free(val_mat)

            global_intercept = global_intercept + intercept_free
            df_pair_lookup_set[pair] = pair_xy_free
            df_lookup1 = pd.DataFrame(main_x_free, columns = [var1]).copy()
            df_lookup2 = pd.DataFrame(main_y_free, columns = [var2]).copy()

            if var1 in df_single_lookup_set.keys():
                df_single_lookup_set[var1] = combine_lookups(df_single_lookup_set[var1], df_lookup1)
            else:
                df_single_lookup_set[var1] = df_lookup1
            if var2 in df_single_lookup_set.keys():
                df_single_lookup_set[var2] = combine_lookups(df_single_lookup_set[var2], df_lookup2)
            else:
                df_single_lookup_set[var2] = df_lookup2
    

    elif (decomp_method == "ind")|(decomp_method == "full"):
        df_single_lookup_num_obs, _, df_pair_lookup_num_obs, _ = lookup_obs_helper(X, df_single_lookup_set, df_pair_lookup_set)

        if decomp_method == "ind":
            #create a global intercept by adding averages from tables -- single effect
            for single in list(df_single_lookup_set.keys()):
                val_mat = df_single_lookup_set[single].copy()
                freqs = df_single_lookup_num_obs[single]
                dist = freqs / freqs.sum()
                single_intercept = (val_mat * dist).sum().values[0]
                df_single_new = val_mat - single_intercept
                global_intercept = global_intercept + single_intercept
                df_single_lookup_set[single] = df_single_new
            
            # create a global intercept by adding averages from tables -- pairwise effect
            for pair in list(df_pair_lookup_set.keys()):
                var1, var2 = pair
                val_mat = df_pair_lookup_set[pair].copy()
                freqs = df_pair_lookup_num_obs[pair]
                dist = freqs / freqs.sum().sum()

                intercept_ind, main_x_ind, main_y_ind, pair_xy_ind, max_col_row_mean_ind = decompose_ind(dist, val_mat)

                global_intercept = global_intercept + intercept_ind
                df_pair_lookup_set[pair] = pair_xy_ind
                df_lookup1 = pd.DataFrame(main_x_ind, columns = [var1]).copy()
                df_lookup2 = pd.DataFrame(main_y_ind, columns = [var2]).copy()

                if var1 in df_single_lookup_set.keys():
                    df_single_lookup_set[var1] = combine_lookups(df_single_lookup_set[var1], df_lookup1)
                else:
                    df_single_lookup_set[var1] = df_lookup1
                if var2 in df_single_lookup_set.keys():
                    df_single_lookup_set[var2] = combine_lookups(df_single_lookup_set[var2], df_lookup2)
                else:
                    df_single_lookup_set[var2] = df_lookup2
        else:
            #create a global intercept by adding averages from tables -- single effect
            for single in list(df_single_lookup_set.keys()):
                val_mat = df_single_lookup_set[single].copy()
                freqs = df_single_lookup_num_obs[single]
                dist = freqs / freqs.sum()
                single_intercept = (val_mat * dist).sum().values[0]
                df_single_new = val_mat - single_intercept
                global_intercept = global_intercept + single_intercept
                df_single_lookup_set[single] = df_single_new
            
            # create a global intercept by adding averages from tables -- pairwise effect
            for pair in list(df_pair_lookup_set.keys()):
                var1, var2 = pair
                val_mat = df_pair_lookup_set[pair].copy()
                freqs = df_pair_lookup_num_obs[pair]
                dist = freqs / freqs.sum().sum()

                intercept_full, main_x_full, main_y_full, pair_xy_full, max_col_row_mean_full = decompose_full(dist, val_mat)

                global_intercept = global_intercept + intercept_full
                df_pair_lookup_set[pair] = pair_xy_full
                df_lookup1 = pd.DataFrame(main_x_full, columns= [var1]).copy()
                df_lookup2 = pd.DataFrame(main_y_full, columns= [var2]).copy()

                if var1 in df_single_lookup_set.keys():
                    df_single_lookup_set[var1] = combine_lookups(df_single_lookup_set[var1], df_lookup1)
                else:
                    df_single_lookup_set[var1] = df_lookup1
                if var2 in df_single_lookup_set.keys():
                    df_single_lookup_set[var2] = combine_lookups(df_single_lookup_set[var2], df_lookup2)
                else:
                    df_single_lookup_set[var2] = df_lookup2
    
    else:
        print("Please enter free, ind, full as the decomposition method!")
    
    return global_intercept, df_single_lookup_set, df_pair_lookup_set

def clean_lookup(df_lookup):
    """Remove repeated cells with the same threshold.
    
    Parameters
    -------------
    df_lookup: pandas dataframe
        dataframe of main effect
    """
    overlap_index = [df_lookup.index[i_row] for i_row in range(df_lookup.shape[0] - 1) if df_lookup.iloc[i_row][0] == df_lookup.iloc[i_row + 1][0]]
    return df_lookup.drop(overlap_index)





