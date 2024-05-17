# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:09:34 2024

@author: aikan
"""

import numpy as np
import pandas as pd
import Model_MSM as MSM
from scipy.optimize import minimize
import yfinance as yf

def calculate_empirical_distribution(data):
    
    Fy = np.arange(1, len(data) + 1) / len(data)

    fy = np.diff(np.concatenate([[0], Fy])) / np.diff(np.concatenate([[0], data.sort_values().values]))

    return Fy, fy

def plackett_cdf(u, v, param):
    cnorm = param-1
    numerator = 1+cnorm*(u+v) - np.sqrt((1+cnorm*(u+v))**2-4*param*cnorm*u*v)
    return numerator/(2*cnorm)

def clayton_cdf(u, v, param):
    expo = -param
    return (u**expo + v**expo - 1)**(1/expo)

def rotated_clayton_cdf(u, v, param):
    return u + v - 1 + clayton_cdf(1-u, 1-v, param)

def joe_clayton_cdf(u, v, param):
    t_low = param[0]
    t_up = param[1]
    k = 1/np.log2(2-t_up)
    g = -1/np.log2(t_low)
    return 1 - (1 - ((1 - (1 - u)**k)**(-g) + (1 - (1 - v)**k)**(-g) - 1)**(-1/g))**(1/k)

def sjc_cdf(u, v, param):
    return 0.5*(joe_clayton_cdf(u, v, param) + joe_clayton_cdf(u, v, param[::-1]+u+v-1))

def frank_cdf(u, v, param):
    coef = lambda x : 1-np.exp(-param*x)
    return -np.log(1-coef(u)*coef(v)/coef(1))/param

def gumbel_cdf(u, v, param):
    return np.exp(-((-np.log(u))**param + (-np.log(v))**param)**(1/param))

def rotated_gumbel_cdf(u, v, param):
    return u + v - 1 + gumbel_cdf(1-u, 1-v, param)

def mixed_derivative(func,u,v,param,h=1e-5):
    return (func(u + h, v + h, param) - func(u + h, v - h, param) - func(u - h, v + h, param) + func(u - h, v - h, param)) / (4 * h * h)

def copula_pdf(copula_cdf, u, v, param, h=1e-5):
    return mixed_derivative(copula_cdf, u, v, param, h)

def copula_likelihood(param, copula_cdf, f1, f2, F1, F2, h=1e-5) :
    ll = 0
    for i in range(len(f1)):
        c = max(copula_pdf(copula_cdf, F1[i], F2[i], param, h), 1e-20)
        ll -= np.log(c) + np.log(f1[i]) + np.log(f2[i])
    return ll 

def AIC(k, ll):
    return 2*(k-ll)

def BIC(k, ll, N):
    return -2*ll+k*np.log(N)

def optimize(copula_cdf, f1, f2, F1, F2, param, bounds):
    result = minimize(copula_likelihood, param, args=(copula_cdf, f1, f2, F1, F2),
                      bounds=bounds, method='L-BFGS-B')
    optimal_param = result.x[0]
    min_log_likelihood = result.fun
    return optimal_param, min_log_likelihood

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
    N = len(fy_nasdaq)
    print('')
    
    # Plackett
    
    param = 10
    bounds = [(1e-20, np.inf)]
    optimal_param, min_log_likelihood = optimize(plackett_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("Plackett copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(1, -min_log_likelihood)}, BIC: {BIC(1, -min_log_likelihood, N)}")
    print('')
    
    # Clayton
    
    param = 5
    bounds = [(1e-20, np.inf)]
    optimal_param, min_log_likelihood = optimize(clayton_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("Clayton copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(1, -min_log_likelihood)}, BIC: {BIC(1, -min_log_likelihood, N)}")
    print('')
    
    # Rotated Clayton
    
    param = 5
    bounds = [(1e-20, np.inf)]
    optimal_param, min_log_likelihood = optimize(rotated_clayton_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("Rotated Clayton copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(1, -min_log_likelihood)}, BIC: {BIC(1, -min_log_likelihood, N)}")
    print('')
    
    # SJC
    
    param = [0.5, 0.5]
    bounds = [(1e-20, 0.99), (1e-20, 0.99)]
    optimal_param, min_log_likelihood = optimize(sjc_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("SJC copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(2, -min_log_likelihood)}, BIC: {BIC(2, -min_log_likelihood, N)}")
    print('')
    
    # Frank
    
    param = 20
    bounds = [(1e-20, np.inf)]
    optimal_param, min_log_likelihood = optimize(frank_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("Frank copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(1, -min_log_likelihood)}, BIC: {BIC(1, -min_log_likelihood, N)}")
    print('')
    
    # Gumbel
    
    param = 5
    bounds = [(1+1e-20, np.inf)]
    optimal_param, min_log_likelihood = optimize(gumbel_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("Gumbel copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(1, -min_log_likelihood)}, BIC: {BIC(1, -min_log_likelihood, N)}")
    print('')
    
    # Rotated Gumbel
    
    param = 5
    bounds = [(1+1e-20, np.inf)]
    optimal_param, min_log_likelihood = optimize(rotated_gumbel_cdf, fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, param, bounds)
    print("Rotated Gumbel copula")
    print(f"Optimal parameter: {optimal_param}, Minimum log-likelihood: {-min_log_likelihood}")
    print(f"AIC: {AIC(1, -min_log_likelihood)}, BIC: {BIC(1, -min_log_likelihood, N)}")
    print('')
    
    