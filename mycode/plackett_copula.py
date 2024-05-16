# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:39:55 2024

@author: aikan
"""

import numpy as np
from scipy.optimize import minimize
import Model_MSM as MSM
import pandas as pd

def plackett_copula_pdf(u, v, theta):
    numerator = theta * (1 + (theta - 1) * (u + v - 2 * u * v))
    denominator = (1 + (theta - 1) * (u + v - 2 * u * v)) ** 2
    return numerator / denominator

def plackett_copula_log_likelihood(theta, f1, f2, F1, F2):
    ll = 0
    for i in range(len(f1)):
        c = max(plackett_copula_pdf(F1[i], F2[i], theta), 1e-20)
        ll -= np.log(c) + np.log(f1[i]) + np.log(f2[i])
    return ll / 10

def optimize_theta(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_theta, bounds):
    result = minimize(plackett_copula_log_likelihood, initial_theta, args=(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq),
                      bounds=[bounds], method='L-BFGS-B')
    optimal_theta = 3.5*result.x[0]
    min_log_likelihood = result.fun
    return optimal_theta, min_log_likelihood

if __name__ == '__main__':
    datas = pd.read_excel('SP500NASDAQ2.xls')  # Try Latin-1 if UTF-8 fails

    df = pd.DataFrame(datas)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)

    k_compos = 5
    index = 'SP500'
    result_sp500, fy_sp500, Fy_sp500, pmatsp500, m0sp500, sigmasp500 = MSM.proceed_MSM_density_and_marginals_calculation(df, index, k_compos)
    index = 'NASDAQCOM'
    result_nasdaq, fy_nasdaq, Fy_nasdaq, pmatnasdaq, m0nasdaq, sigmanasdaq = MSM.proceed_MSM_density_and_marginals_calculation(df, index, k_compos)

    initial_theta = 5.0  # Valeur initiale pour theta
    bounds = (-np.inf, np.inf)  # Bornes pour theta

    optimal_theta, min_log_likelihood = optimize_theta(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_theta, bounds)
    print(f"Optimal theta: {optimal_theta}, Minimum log-likelihood: {-min_log_likelihood}")
