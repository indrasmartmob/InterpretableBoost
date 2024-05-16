import pandas as pd
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps, ticker, gridspec
import json
from xgboost import XGBRegressor, XGBClassifier

def expit(x):
    """
    Sigmoid transformation: 1/(1+e^(-eta))
    """
    return 1/(1+np.exp(-x))

def logit(x):
    """
    Logit transformation: log(p/(1-p))
    """
    return np.log(x/(1-x))


