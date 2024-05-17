import pandas as pd
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps, ticker, gridspec
import json
from xgboost import XGBRegressor, XGBClassifier

from ..logfile import logger, log_enable, log_disable

def fn_check(verbose=True):
    if verbose==True:
        log_enable()
    elif verbose==False:
        log_disable()
    else :
        log_disable()

    logger.info("In the fn_check method")
    logger.trace("A trace message.")
    logger.debug("A debug message.")
    logger.info("An info message.")
    logger.success("A success message.")
    logger.warning("A warning message.")
    logger.error("An error message.")
    logger.critical("A critical message.")
    log_disable()
    return None

def fn_sigmoid(x):
    """
    Sigmoid transformation: 1/(1+e^(-eta))
    """
    return 1/(1+np.exp(-x))

def fn_logit(x):
    """
    Logit transformation: log(p/(1-p))
    """
    return np.log(x/(1-x))

def fn_combine_lookups(df1, df2):
    """
    
    """
    logger.info(F"Starting of {fn_combine_lookups.__qualname__}")
    
    df_new1 = df1.copy()
    for threshold in df2.index:
        if threshold not in df1.index:
            val = df1.iloc[sum(df1.index < threshold)]
            val.name = threshold
            df_new1 = df_new1.append(val)
    df_new1 = df_new1.sort_index()

    
    df_new2 = df2.copy()
    for threshold in df1.index:
        if threshold not in df2.index:
            val = df2.iloc[sum(df2.index < threshold)]
            val.name = threshold
            df_new2 = df_new2.append(val)
    df_new2 = df_new2.sort_index()
    


    logger.info(F"Ending of {fn_combine_lookups.__qualname__}")
    return df_new1 + df_new2

def fn_clean_lookup(df):
    """
    """
    overlap_index = [df.index[i_row] for i_row in range(df.shape[0]-1) if df.iloc[i_row][0] == df.iloc[i_row + 1][0]]

    return df.drop(overlap_index)




