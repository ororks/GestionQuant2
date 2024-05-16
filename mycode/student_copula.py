# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:41:58 2024

@author: aikan
"""
import numpy as np
from scipy.stats import t, multivariate_t
from scipy.optimize import minimize
import pandas as pd
import Model_MSM as MSM

def student_copula_pdf(u, v, rho, nu):
    # Inverse CDF (quantile function) of the univariate t-distribution
    t_inv_u = t.ppf(u, df=nu)
    t_inv_v = t.ppf(v, df=nu)
    
    # PDF of the univariate t-distribution
    pdf_t_inv_u = t.pdf(t_inv_u, df=nu)
    pdf_t_inv_v = t.pdf(t_inv_v, df=nu)
    
    # Mean and covariance matrix for the bivariate t-distribution
    mean = [0, 0]
    cov = np.array([[1, rho], [rho, 1]])
    
    # Ensure the covariance matrix is positive definite
    if np.linalg.det(cov) <= 0:
        return 1e-20  # Return a small value to avoid errors
    
    # PDF of the bivariate t-distribution
    if np.isscalar(u):
        pdf_bivariate_t = multivariate_t.pdf([t_inv_u, t_inv_v], mean, cov, df=nu)
    else:
        pdf_bivariate_t = multivariate_t.pdf(np.column_stack((t_inv_u, t_inv_v)), mean, cov, df=nu)
    
    # Copula density
    copula_density = pdf_bivariate_t / (pdf_t_inv_u * pdf_t_inv_v)
    return copula_density

def Student_Copula_Pdf(u, v, param):
    return student_copula_pdf(u, v, param[0], param[1])

def student_copula_LL(params, f1, f2, F1, F2):
    rho, nu = params
    ll = 0
    for i in range(len(f1)):
        c = max(student_copula_pdf(F1[i], F2[i], rho, nu), 1e-20)
        ll -= np.log(c) + np.log(f1[i]) + np.log(f2[i])
    return ll/10

# Define the optimization routine
def optimize_theta_and_nu(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_params, bounds):
    # Minimize negative log-likelihood to find optimal rho and nu
    result = minimize(student_copula_LL, initial_params, args=(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq),
                      bounds=bounds, method='L-BFGS-B')
    
    # Optimal values of rho and nu
    optimal_rho = result.x[0]
    optimal_nu = result.x[1]
    
    # Minimum log-likelihood value
    min_log_likelihood = result.fun
    
    return optimal_rho, optimal_nu, min_log_likelihood

# Example usage:
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

    # Initial values for rho and nu
    initial_params = [0.5, 5.0]
    bounds = [(-0.99, 0.99), (2, 30)]  # Bounds for rho and nu

    optimal_rho, optimal_nu, min_log_likelihood = optimize_theta_and_nu(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_params, bounds)
    print(f"Optimal rho: {optimal_rho}, Optimal nu: {optimal_nu}, Minimum log-likelihood: {-min_log_likelihood}")

    