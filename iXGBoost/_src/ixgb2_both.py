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

def convert_row(data_col, cutoffs):
    """Assigns main effect values.
    
    Parameters
    -------------
    data_col: pandas series
        series of data column
    cutoffs: pandas dataframe
        lookup table of main effect
    """
    if np.isnan(cutoffs.index).any():
        cutoffs_temp = cutoffs.drop(labels=[np.nan])
    else:
        cutoffs_temp = cutoffs
    # Get the intervals used for cutoff values
    intervals = [-np.inf] + [i for i in cutoffs_temp.index.to_list()]
    intervals = sorted(list(set(intervals)))

    # Generate the mapping dictionary
    k = np.arange(len(intervals)-1)
    v = cutoffs_temp.values.ravel()
    map_dict = dict(map(lambda i, j: (i, j), k, v))
    if np.isnan(cutoffs.index).any():
        map_dict[np.nan] = cutoffs.loc[np.nan, :].values[0]
    
    # Get the indices for the binned values
    values_bin = pd.cut(data_col, intervals, labels=False, right=False)

    return values_bin.map(map_dict).values

def convert_row_2d(data_cols1, data_cols2, cutoffs):
    """Assigns pairwise interaction effect values.
    
    Parameters
    -------------
    data_cols1: pandas series
        series of data column
    data_cols2: pandas series
        series of data column
    cutoffs: pandas dataframe
        lookup table of pairwise interaction effect
    """
    if np.isnan(cutoffs.index).any():
        cutoffs_temp = cutoffs.drop(np.nan, axis=1).drop(np.nan, axis=0)
    else:
        cutoffs_temp = cutoffs
    intervals1 = [-np.inf] + cutoffs_temp.columns.to_list()
    intervals2 = [-np.inf] + cutoffs_temp.index.to_list()
    intervals1 = sorted(list(set(intervals1)))
    intervals2 = sorted(list(set(intervals2)))

    map_dict = {}
    for idx_i, v in enumerate(cutoffs_temp.values):
        for idx_j, w in enumerate(v):
            map_dict.update({(idx_i, idx_j): w})
    
    if np.isnan(cutoffs.index).any():
        for ind_z, z in enumerate(cutoffs.loc[np.nan, :].values[1:]):
            map_dict.update({("NaN", ind_z): z})
        for ind_u, u in enumerate(cutoffs.loc[:, np.nan].values[1:]):
            map_dict.update({(ind_u, "NaN"): u})
        map_dict.update({("NaN", "NaN"): cutoffs.loc[np.nan, np.nan]})

        c1 = pd.cut(data_cols1, intervals1, labels=False, right=False).fillna("NaN")
        c2 = pd.cut(data_cols2, intervals2, labels=False, right=False).fillna("NaN")
    else:
        c1 = pd.cut(data_cols1, intervals1, labels=False, right=False)
        c2 = pd.cut(data_cols2, intervals2, labels=False, right=False)        

    return pd.Series(map(lambda x: map_dict[x], zip(c2, c1))).values

def predict_effects_helper(X, single_effect_index, df_single_lookup_set, pair_effect_index, df_pair_lookup_set):
    """Assigns main effect values and pairwise effect values to input data, and provides the prediction score
    
    Parameters
    --------------
    X: pandas dataframe
        input data
    single_effect_index: list
        a list of names of main effect
    pair_effect_index: list
        a list of names of pairwise interaction effect
    df_single_lookup_set: pandas dataframe
        lookup table for main effect
    df_pair_lookup_set: pandas dataframe
        lookup table for pairwise interactoin effect
    
    """

    all_results = []
    all_col_names = []

    for idx, col_name in enumerate(single_effect_index):
        _t = convert_row(X[col_name], df_single_lookup_set[col_name])
        all_results.append(_t)
        all_col_names.append(col_name)
    
    for idx, col_names in enumerate(pair_effect_index):
        _t = convert_row_2d(X[col_names[0]], X[col_names[1]], df_pair_lookup_set[col_names])
        all_results.append(_t)
        all_col_names.append(tuple((col_names[0], col_names[1])))
    
    effects = pd.DataFrame(
        dict(map(lambda i, j: (i, j), all_col_names, all_results)),
    )

    return effects

def convert_lookup(data_col, cutoff):
    """Assigns main effect values when there are missing values.
    
    Parameters
    -------------
    data_cols: pandas series
        series of data column
    cutoffs: pandas dataframe
        lookup table of main effect
    """
    cutoffs = cutoff.copy()
    if np.isnan(cutoffs.index).any():
        cutoffs_temp = cutoffs.drop(labels=[np.nan])
    else:
        cutoffs_temp = cutoffs
    # Get the intervals used for cutoff values
    intervals = [-np.inf] + [i for i in cutoffs_temp.index.to_list()]
    intervals = sorted(list(set(intervals)))

    map_dict = {cutoffs_temp.index[i]: set() for i in range(cutoffs_temp.shape[0])}
    if np.isnan(cutoffs.index).any():
        map_dict["NaN"] = set()
        values_bin = pd.cut(data_col, intervals, labels=False, right=False).fillna("NaN")
    else:
        values_bin = pd.cut(data_col, intervals, labels=False, right=False)
    
    for idx, val in zip(values_bin.index, values_bin.values):
        if val == "NaN":
            map_dict["NaN"].add(idx)
        else:
            map_dict[cutoffs_temp.index[val]].add(idx)

    new_lookup = pd.DataFrame(list(map_dict.items()), columns = [data_col.name, "obs_location"])
    new_lookup = new_lookup.set_index(new_lookup.columns[0]).iloc[:, -1]

    for keys, values in map_dict.items():
        if keys == "NaN":
            cutoffs.loc[np.nan]=len(values)
        else:
            cutoffs.loc[cutoffs.index==keys]=len(values)
    
    return cutoffs, new_lookup

def convert_lookup_2d(data_cols1, data_cols2, cutoff):
    """Assigns pairwise ineraction effect values when there are missing values.
    
    Parameters
    ------------
    data_cols1: pandas series
        series of data column
    data_cols2: pandas series
        series of data column
    cutoffs: pandas dataframe
        lookup table of pairwise interaction effect
    """
    cutoffs = cutoff.copy()
    # Get the intervals used for cutoff values
    if np.isnan(cutoffs.index).any():
        cutoffs_temp = cutoffs.drop(np.nan, axis=1).drop(np.nan, axis=0)
    else:
        cutoffs_temp = cutoffs
    intervals1 = [-np.inf] + cutoffs_temp.columns.to_list()
    intervals2 = [-np.inf] + cutoffs_temp.index.to_list()
    intervals1 = sorted(list(set(intervals1)))
    intervals2 = sorted(list(set(intervals2)))

    pair_set = []
    for row in cutoffs_temp.index:
        for col in cutoffs_temp.columns:
            pair_set.append(tuple((row, col)))
    map_dict = {tuple(var): set() for var in pair_set}

    if np.isnan(cutoffs.index).any():
        for z in cutoffs.columns[1:]:
            map_dict.update({("NaN", z): set()})
        for u in cutoffs.index[1:]:
            map_dict.update({(u, "NaN"): set()})   
        map_dict.update({("NaN", "NaN"): set()})
        c1 = pd.cut(data_cols1, intervals1, labels=False, right=False).fillna("NaN")
        c2 = pd.cut(data_cols2, intervals2, labels=False, right=False).fillna("NaN")
    else:
        c1 = pd.cut(data_cols1, intervals1, labels=False, right=False)
        c2 = pd.cut(data_cols2, intervals2, labels=False, right=False)
    
    for idx, val2, val1 in zip(c2.index, c2.values, c1.values):
        if (val2 == "NaN")&(val1 != "NaN"):
            map_dict[tuple(("NaN", cutoffs_temp.columns[val1]))].add(idx)
        elif (val1 == "NaN")&(val2 != "NaN"):
            map_dict[tuple((cutoffs_temp.index[val2], "NaN"))].add(idx)
        elif (val2 == "NaN")&(val1 == "NaN"):
            map_dict[tuple(("NaN", "NaN"))].add(idx)
        else:
            map_dict[tuple((cutoffs_temp.index[val2], cutoffs_temp.columns[val1]))].add(idx)
    
    new_lookup = pd.DataFrame(list(map_dict.items()), columns = [tuple((data_cols1.name, data_cols2.name)), "obs_location"])
    new_lookup = new_lookup.set_index(new_lookup.columns[0]).iloc[:, -1]

    for keys, values in map_dict.items():
        if (keys[0] != "NaN")&(keys[1] == "NaN"):
            cutoffs.loc[cutoffs.index==keys[0], np.nan]=len(values)
        elif (keys[0] == "NaN")&(keys[1] != "NaN"):
            cutoffs.loc[np.nan, cutoffs.columns==keys[1]]=len(values)
        elif (keys[0] == "NaN")&(keys[1] == "NaN"):
            cutoffs.loc[np.nan, np.nan]=len(values)
        else:
            cutoffs.loc[cutoffs.index==keys[0], cutoffs.columns==keys[1]]=len(values)
    
    return cutoffs, new_lookup

def lookup_obs_helper(X, df_single_lookup_set, df_pair_lookup_set):

    """calculate # of observations in each cell in lookup tables and assign locations of observations in each lookup tables
    
    Parameters
    --------------
    X: pandas dataframe
        input data
    df_single_lookup_set: pandas dataframe
        lookup table for main effect
    df_pair_lookup_set: pandas dataframe
        lookup table for pairwise interaction effect

    """

    df_pair_lookup_num_obs = dict()
    df_pair_lookup_obs_location = dict()
    single_effect_index = list(df_single_lookup_set.keys())
    pair_effect_index = list(df_pair_lookup_set.keys())

    for idx, col_names in enumerate(pair_effect_index):
        _table2d1, _table2d2 = convert_lookup_2d(X[col_names[0]], X[col_names[1]], df_pair_lookup_set[col_names])
        df_pair_lookup_num_obs[col_names] = _table2d1
        df_pair_lookup_obs_location[col_names] = _table2d2
    
    df_single_lookup_num_obs = dict()
    df_single_lookup_obs_location = dict()
    for idx, col_name in enumerate(single_effect_index):
        _table1, _table2 = convert_lookup(X[col_name], df_single_lookup_set[col_name])
        df_single_lookup_num_obs[col_name] = _table1
        df_single_lookup_obs_location[col_name] = _table2
    
    return df_single_lookup_num_obs, df_single_lookup_obs_location, df_pair_lookup_num_obs, df_pair_lookup_obs_location

def get_lookup_table_helper(effect_index, pair_effect_index, df_pair_lookup_set, single_effect_index, df_single_lookup_set):

    """get the lookup table for a specific effect
    
    Parameters
    -------------
    effect_index: string
        the name of effect
    single_effect_index: list
        a list of names of main effect
    pair_effect_index: list
        a list of names of pairwise interaction effect
    df_single_lookup_set: pandas dataframe
        lookup table for main effect
    df_pair_lookup_set: pandas dataframe
        lookup table for pairwise interaction effect

    """

    if (type(effect_index) is tuple)&(len(effect_index)==2):
        if effect_index in pair_effect_index:
            var1, var2 = effect_index
            print("Pairwise effect lookup table (logodds) for features:" + var1 + " (column)" + " and " + var2 + " (row)")
            return df_pair_lookup_set[effect_index]
        
        elif effect_index[::-1] in pair_effect_index:
            var1, var2 = effect_index
            print("Pairwise effect lookup table (logodds) for features:" + var1 + " (column)" + " and " + var2 + " (row)")
            return df_pair_lookup_set[effect_index[::-1]].T
        
        else:
            print("This pairwise effect does not exist!")
            return
    
    elif type(effect_index) is str:
        print("Single effect lookup table (logodds) for feature: " + effect_index)
        if effect_index in single_effect_index:
            return df_single_lookup_set[effect_index]
        
        else:
            print("This single effect does not exist!")
            return
    
    else:
        print("Please enter tuple with length of 2 for pairwise effect or single string for single effect!")
        return

def main_val_plot(gsi, f, var, X, int_var_set, lookup_set, num_bins, effect_imp_var, bound_pct, bound_val, verbose):
    """Plot main effect.
    
    Parameters
    --------------
    var: string
        name of the feature for plotting
    X: pandas dataframe
        dataset for histogram
    int_var_set: list
        list of names of integer values features
    lookup_set: dictionary
        dictionary of main effect lookup tables
    num_bins: integer
        number of bins for histogram
    effect_imp_var: float
        effect importance
    bound_pct: list of float
        lower bound percentile and upper bound percentile for clipping
    bound_val: list of float
        lower bound value and upper bound value for clipping
    """
    gs_i = gsi.subgridspec(20, 1, wspace=0.7, hspace=1.4)
    ax_main = f.add_subplot(gs_i[:17, 0])

    x = X[var]
    x_min = np.min(x)
    x_max = np.max(x)

    if (var in int_var_set) & (x_max - x_min < 30):
        index = True
        if verbose:
            print(f"Feature: {var}")
            print(f"min={np.min(x)}, max={np.max(x)}")
            print(f"This feature is integer, no need to be capped!")
        df_x_val = x.value_counts()
        df_table = lookup_set

        t = df_x_val.index.tolist()
        apart = t[1] - t[0]
        tn = [t[0] - apart] + t
    
    else:
        index = False
        if len(bound_pct) == 2:
            lb_pct, ub_pct = bound_pct
            wq = DescrStatsW(data=np.array(X[var]))
            [x_min, x_max] = wq.quantile(probs=np.array([lb_pct, ub_pct]), return_pandas=False)
        if len(bound_val) == 2:
            x_min, x_max = bound_val
        x = X.loc[(((X[var]>=x_min)&(X[var]<=x_max))|(X[var].isna())), var]
        if verbose:
            print(f"Feature: {var}")
            print(f"min={np.min(X[var])}, max={np.max(X[var])}")
            print(f"capped at: lb={x_min}, ub={x_max}")
        df_x_val = x.value_counts()
        df_table = lookup_set
        df_table = df_table.loc[((df_table.index>=x_min)&(df_table.index<=x_max))|(df_table.index.isna())]

        xx = np.linspace(x_min, x_max, num=num_bins)
        t = list(xx)
        apart = t[1] - t[0]
        tn = [t[0]-apart]+t

    y_min = np.min(df_table.values)
    y_max = np.max(df_table.values)


    ### plot main effect
    ax_main.set_title(str(var) + " (" + str("{0:.1%}".format(effect_imp_var)) + ')', fontsize=10)
    plt.ylabel("log odds")

    plt.xlim([x_min - 2* apart, x_max])
    plt.xticks([])

    if var in int_var_set:
        df_table.index = np.ceil(np.array(df_table.index)) - 0.5
        plt.xlim([x_min - 0.5 -2* apart, x_max + 0.5])
    
    x_lb_pt = np.hstack([x_min, np.array(df_table.index[1:-1].tolist()).flatten(), x_max + 0.5])
    try:
        y_lb_pt = np.hstack([df_table.values[1], np.array(df_table.values[1:].tolist()).flatten()])
    except:
        print(f"For feature {var}, you clipped too much data! Please increase upper bound or decrease the lower bound!")
    else:
        plt.step(x_lb_pt, y_lb_pt)
    
    # if X.isnull().values.any():
    x_nan = tn[0]
    y_nan = df_table.values[0]
    plt.plot(x_nan, y_nan, "o", color="r")

    ### plot histogram or bar plot
    ax_bottom = f.add_subplots(gs_i[17:, 0])

    if index:
        ax_bottom.bar(t, df_x_val.tolist(), color="b", alpha=0.1, width=apart, edgecolor="b")
        tlabels=[str(int(e) for e in t)]
        ax_bottom.set_xticks(t)
        ax_bottom.set_xticklabels(tlabels, rotation=45)
    
        # if X.isnull().values.any():
        ax_bottom.bar([x_nan], [sum(x.isna())], color="r", alpha=0.1, width=apart, edgecolor="r")
        tlabels=[str(int(e)) for e in tn]
        tlabels[0]="NaN"
        ax_bottom.set_xticks(tn)
        ax_bottom.set_xticklabels(tlabels, rotation=45)

    else:
        ax_bottom.hist(x, bins= num_bins, alpha=0.1, color="b", edgecolor="b")
        tlabels=["{:.1f}".format(e) for e in tn]
        ax_bottom.set_xticks(t)
        ax_bottom.set_xticklabels(tlabels, rotation=45)

        # if X.isnull().values.any():
        ax_bottom.bar([x_nan], [sum(x.isna())], color="r", alpha=0.1, width=apart, edgecolor="r")
        tlabels=["{:.1f}".format(e) for e in tn]
        tlabels[0]="NaN"
        ax_bottom.set_xticks(tn)
        ax_bottom.set_xticklabels(tlabels, rotation=45)

    if var in int_var_set:
        plt.xlim([x_min - 0.5 - 2*apart, x_max + 0.5])
    else:
        plt.xlim([x_min - 2*apart, x_max])
    
    plt.yticks([])

def pairwise_val_plot(gsi, f, pair, X, int_var_set, lookup_set, effect_imp_var, bound_pct, bound_val, verbose):
    """Plot pairwise interaction efect.
    
    Parameters
    ------------
    pair: tuple
        tuple of names of two features for plotting
    X: pandas dataframe
        dataset for histogram
    int_var_set: list
        list of names of integer values features
    lookup_set: dictionary
        dictionary of pairwise interaction effect lookup tables
    effect_imp_var: float
        effect importance
    bound_pct: list of list of float
        lower bound percentile and upper bound percentile for clipping
    bound_val: list of lilst of float
        lower bound value and upper bound value for clipping
    """

    ### plot pairwise interaction effect
    var1, var2 = pair
    bound_pct1, bound_pct2 = bound_pct
    bound_val1, bound_val2 = bound_val
    if verbose:
        print(f"Feature: {var1} and {var2}")
    
    x = X[var1]
    x_min = np.min(x)
    x_max = np.max(x)
    y = X[var2]
    y_min = np.min(y)
    y_max = np.max(y)

    if var1 in int_var_set:
        if verbose:
            print(f"var1: min={x_min}, max={x_max}")
            print(f"This feature is integer, no need to be capped!")
    else:
        if len(bound_pct1) == 2:
            lb_pct1, ub_pct1 = bound_pct1
            wq1 = DescrStatsW(data=np.array(X[var1]))
            [x_min, x_max] = wq1.quantile(probs=np.array([lb_pct1, ub_pct1]), return_pandas=False)
        if len(bound_val1) == 2:
            x_min, x_max = bound_val1
        x = X.loc[(X[var1]>=x_min)&(X[var1]<=x_max), var1]
        if verbose:
            print(f"var1: min={np.min(X[var1])}, max={np.max(X[var1])}")
            print(f"var1 capped at: lb={x_min}, ub={x_max}")
        
    if var2 in int_var_set:
        if verbose:
            print(F"var2: min={y_min}, max={y_max}")
            print(f"This feature is integer, no need to be capped!")
    else:
        if len(bound_pct2) == 2:
            lb_pct2, ub_pct2 = bound_pct2
            wq2 = DescrStatsW(data=np.array(X[var2]))
            [y_min, y_max] = wq2.quantile(probs=np.array([lb_pct2, ub_pct2]), return_pandas=False)
        if len(bound_val2) == 2:
            y_min, y_max = bound_val2
        y = X.loc[(X[var2]>=y_min)&(X[var2]<=y_max), var2]
        if verbose:
            print(f"var2: min={np.min(X[var2])}, max={np.max(X[var2])}")
            print(F"var2 capped at: lb={y_min}, ub={y_max}")
    
    df_x_val = x.value_counts()
    df_y_val = y.value_counts()

    gs_i = gsi.subgridspec(20, 20, wspace=0.7, hspace=1.4)

    ax_main = f.add_subplot(gs_i[:17, 3:-1])

    ax_main.set_title(str(var1) + " vs "+ str(var2) + ' (' + str("{0:.1%}".format(effect_imp_var)) + ')', fontsize=11)

    df_table_org = lookup_set

    val_min = np.min(df_table_org.values)
    val_max = np.max(df_table_org.values)

    df_table = (df_table_org - val_min)/(val_max - val_min)


    if var1 in int_var_set:
        df_table.columns = np.ceil(np.array(df_table.columns)) - 0.5
        x_lb_pt = np.hstack([x_min - 0.5, np.array(df_table.columns[:-1])])
        x_len = np.hstack([np.array(df_table.columns[:-1]), x_max + 0.5]) - np.hstack([x_min - 0.5, np.array(df_table.columns[:-1])])
    else:
        x_lb_pt = np.hstack([x_min, np.array(df_table.columns[:-1])])
        x_len = np.hstack([np.array(df_table.columns[:-1]), x_max]) - np.hstack([x_min, np.array(df_table.columns[:-1])])
    if var1 in int_var_set:
        plt.xlim([x_min - 0.5, x_max + 0.5])
    else:
        plt.xlim([x_min, x_max])
        # if (var2 in int_var_set) & (y_max - y_min < 30):
    if var2 in int_var_set:
        df_table.index = np.ceil(np.array(df_table.index)) - 0.5
        y_lb_pt = np.hstack([y_min - 0.5, np.array(df_table.index[:-1])])
        y_len = np.hstack([np.array(df_table.index[:-1]), y_max + 0.5]) - np.hstack([y_min - 0.5, np.array(df_table.index[:-1])])
    else:
        y_lb_pt = np.hstack([y_min, np.array(df_table.index[:-1])])
        y_len = np.hstack([np.array(df_table.index[:-1]), y_max]) - np.hstack([y_min, np.array(df_table.index[:-1])])
    if var2 in int_var_set:
        plt.ylim([y_min - 0.5, y_max + 0.5])
    else:
        plt.ylim([y_min, y_max])
    cmap = plt.get_cmap("viridis")
    for i_x in range(len(x_lb_pt)):
        for i_y in range(len(y_lb_pt)):
            rect1 = matplotlib.patches.Rectangle((x_lb_pt[i_x], y_lb_pt[i_y]), x_len[i_x], y_len[i_y],
                                                 color=cmap(df_table.iloc[i_y, i_x]))
            ax_main.add_patch(rect1)
    
    plt.xticks([])
    plt.yticks([])

    ### plot histogram or bar plot for var2 (y)
    ax_left = f.add_subplots(gs_i[:17, 0:3])
    if (var2 in int_var_set) & (y_max - y_min < 30):
        ax_left.barh(df_y_val.index.tolist(), df_y_val.tolist(), height = 1)
        if (y_max - y_min <= 12):
            plt.yticks(df_y_val.index.tolist())
    else:
        ax_left.hist(y, bins = 50, orientation="horizontal", align = "left")
    if var2 in int_var_set:
        plt.ylim([y_min -0.5, y_max + 0.5])
    else:
        plt.ylim([y_min, y_max])
    
    plt.xticks([])

    ### plot histogram or bar plot for var1 (x)
    ax_bottom = f.add_subplots(gs_i[17:, 3:-1])
    if (var1 in int_var_set) & (x_max - x_min < 30):
        ax_bottom.bar(df_x_val.index.tolist(), df_x_val.tolist(), width = 1)
        if (x_max - x_min <= 12):
            plt.xticks(df_x_val.index.tolist())
    else:
        ax_bottom.hist(x, bins = 50, align = "mid")
    if var1 in int_var_set:
        plt.xlim([x_min - 0.5, x_max + 0.50])
    else:
        plt.xlim([x_min, x_max])
    
    plt.yticks([])

    ### plot colormap
    ax_bar = f.add_subplot(gs_i[:17, -1])
    gradient = np.linspace(1, 0, 256)
    gradient = np.vstack((gradient, gradient))
    ax_bar.imshow(gradient.T, extent = [0, 1, val_min, val_max], aspect="auto", cmap=colormaps["viridis"])
    plt.yticks(np.linspace(val_min + (val_max - val_min) / 6, val_max - (val_max - val_min) / 6, num=4))
    ax_bar.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_bar.yaxis.tick_right()
    plt.xticks([])

    
        




