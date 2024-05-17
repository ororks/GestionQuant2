import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize as min
import pandas as pd
import Model_MSM as MSM

def gaussian_copula_log_likelihood(rho, f1, f2, F1, F2):
    ll = 0
    for i in range(len(f1)+1):
        c = max(bivariate_gaussian_copula_pdf(F1[i-1], F2[i-1], rho), 1e-20)
        ll -= np.log(c) + np.log(f1[i-1]) + np.log(f2[i-1])
    return ll/10

# Define joint PDF of standard bivariate normal distribution
def bivariate_normal_pdf(x1, x2, rho):
    z1 = norm.ppf(x1)
    z2 = norm.ppf(x2)
    numerator = np.exp(-(z1**2 - 2 * rho * z1 * z2 + z2**2) / (2 * (1 - rho**2)))
    denominator = np.pi * np.sqrt(1 - rho**2)
    pdf = numerator / denominator
    return pdf

# Define inverse CDF (quantile function) of standard normal distribution
def inv_norm_cdf(u):
    return norm.ppf(u)

# Define PDF of bivariate Gaussian copula
def bivariate_gaussian_copula_pdf(u1, u2, rho):
    x1 = inv_norm_cdf(u1)
    x2 = inv_norm_cdf(u2)
    
    pdf_x1 = norm.pdf(x1)
    pdf_x2 = norm.pdf(x2)
    if pdf_x1 == 0 or pdf_x2 == 0:
        return 0
    else:
        if np.isinf(x1) or np.isinf(x2):
            return 0
        else:
            result = bivariate_normal_pdf(x1, x2, rho) / (pdf_x1 * pdf_x2)
            result = np.nan_to_num(result)
            return result[0]

# Define the optimization routine
def optimize_rho(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_rho, bounds):
    # Minimize negative log-likelihood to find optimal rho
    result = min(gaussian_copula_log_likelihood, initial_rho, args=(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq),
                 bounds=[bounds])

    # Optimal value of rho
    optimal_rho = 2*result.x[0]

    # Minimum log-likelihood value
    min_log_likelihood = result.fun

    return optimal_rho, min_log_likelihood

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
    initial_rho = 10
    bounds = (0,0.99)  # Bounds for rho and nu

    optimal_rho, min_log_likelihood = optimize_rho(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_rho, bounds)
    print(f"Optimal rho: {optimal_rho}, Minimum log-likelihood: {min_log_likelihood}")

    