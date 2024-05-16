# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:27:16 2024

@author: aikan
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

def hansen_spa_test(series):
    """
    Compute the p-value associated with Hansen's Supremum Augmented (SPA) test for heteroskedasticity.
    
    Parameters:
    - series: A time series of residuals from a regression model.
    
    Returns:
    - p_value: The p-value associated with the SPA test.
    """

    X = np.arange(len(series))  
    X = sm.add_constant(X)
    y = series**2
    model = sm.OLS(y, X)
    results = model.fit()
    
    residuals = results.resid
    
    test_statistic = (residuals**2).mean()
    
    degrees_of_freedom = len(series)
    p_value = 1 - chi2.cdf(test_statistic, degrees_of_freedom)
    
    return p_value

if __name__ == '__main__':
    np.random.seed(0)
    num_samples = 1000
    residuals = np.random.normal(0, 1, num_samples)
    p_value = hansen_spa_test(residuals)
    print("P-value associated with Hansen's SPA test:", p_value)
