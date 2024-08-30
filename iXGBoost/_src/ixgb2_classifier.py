import pandas as pd
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps, ticker, gridspec
import json
from xgboost import XGBRegressor, XGBClassifier

from ..logfile import logger, log_enable, log_disable
from .._src import utility

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

class InterpretableXGBClassifier(XGBClassifier):
    """
    Train an interpretable XGBoost Classifier model with monotonocity constraint.

    Parameters
    -----------
    increasing_vars: list
        list of monotonic increasing variables
    decreasing_vars: list
        list of monotonic decreasing variables
    xgb_params: dict
        initial arguments for the XGBoost model
    """

    def __init__(self, increasing_vars = [], decreasing_vars = [], **xgb_params):
        """
        
        """
        log_enable()
        logger.info(F"Starting of __init__:{__name__}")
        super().__init__(**xgb_params)
        self.increasing_vars = increasing_vars
        self.decreasing_vars = decreasing_vars
        logger.info(F"Ending of __init__:{__name__}")
        return None

