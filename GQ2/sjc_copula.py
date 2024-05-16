# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:49:44 2024

@author: aikan
"""
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import rankdata
import Model_MSM as MSM

def sjc_copula_pdf(u, v, theta, delta):
    term1 = (1 - u) ** theta + (1 - v) ** theta - (1 - u) ** theta * (1 - v) ** theta
    term2 = (term1 ** (1 / theta))
    term3 = (1 - term2) ** (delta - 1)
    term4 = term2 ** (1 - delta)
    term5 = (1 - u) ** (theta - 1)
    term6 = (1 - v) ** (theta - 1)
    term7 = theta * delta * (1 - u) ** theta * (1 - v) ** theta

    c = term3 * (term4 + term5 * term6 * term7)

    return c

def sjc_copula_log_likelihood(params, u, v):
    theta, delta = params
    ll = 0
    for i in range(len(u)):
        pdf = sjc_copula_pdf(u[i], v[i], theta, delta)
        c = max(pdf, 1e-20)
        ll -= np.log(c)
    return ll

def optimize_sjc_params(u, v, initial_params, bounds):
    result = minimize(sjc_copula_log_likelihood, initial_params, args=(u, v), bounds=bounds, method='L-BFGS-B')
    optimal_params = result.x
    min_log_likelihood = result.fun
    return optimal_params, min_log_likelihood

if __name__ == '__main__':
    datas = pd.read_excel('SP500NASDAQ2.xls')
    df = pd.DataFrame(datas)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)

    k_compos = 5
    index = 'SP500'
    result_sp500, fy_sp500, Fy_sp500, pmatsp500, m0sp500, sigmasp500 = MSM.proceed_MSM_density_and_marginals_calculation(df, index, k_compos)
    index = 'NASDAQCOM'
    result_nasdaq, fy_nasdaq, Fy_nasdaq, pmatnasdaq, m0nasdaq, sigmanasdaq = MSM.proceed_MSM_density_and_marginals_calculation(df, index, k_compos)

    initial_params = [2.0, 2.0]  # Valeurs initiales pour theta et delta
    bounds = [(1e-05, np.inf), (1e-05, np.inf)]  # Bornes pour theta et delta

    # Normaliser les marges pour obtenir les pseudo-observations u et v
    u = rankdata(Fy_sp500) / (len(Fy_sp500) + 1)
    v = rankdata(Fy_nasdaq) / (len(Fy_nasdaq) + 1)

    optimal_params, min_log_likelihood = optimize_sjc_params(u, v, initial_params, bounds)
    print(f"Optimal parameters: theta={optimal_params[0]}, delta={optimal_params[1]}, Minimum log-likelihood: {-min_log_likelihood}")
