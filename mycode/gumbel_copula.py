# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:33:25 2024

@author: aikan
"""

import numpy as np
from copulas.bivariate import Gumbel
from scipy.optimize import minimize
import Model_MSM as MSM
import pandas as pd

def gumbel_copula_log_likelihood(theta, f1, f2, F1, F2):
    # Crée une instance de la copule de Gumbel avec le paramètre donné theta
    copula = Gumbel()
    copula.theta = theta
    
    ll = 0
    for i in range(len(f1)):
        # Combine F1[i] et F2[i] dans une matrice 2D
        sample = np.array([[F1[i], F2[i]]])
        c = max(copula.pdf(sample), 1e-20)
        ll -= np.log(c) + np.log(f1[i]) + np.log(f2[i])
    return ll / 10

# Définir la routine d'optimisation
def optimize_theta(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_theta, bounds):
    # Minimise la log-vraisemblance négative pour trouver le theta optimal
    result = minimize(gumbel_copula_log_likelihood, initial_theta, args=(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq),
                      bounds=[bounds], method='L-BFGS-B')
    
    # Valeur optimale de theta
    optimal_theta = result.x[0]
    
    # Valeur minimale de la log-vraisemblance
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
