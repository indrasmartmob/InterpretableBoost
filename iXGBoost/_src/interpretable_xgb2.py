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


