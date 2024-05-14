import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize as min

def gaussian_copula_log_likelihood(rho, f1, f2, F1, F2):
    ll = 0
    for i in range(len(f1)+1):
        c = max(bivariate_gaussian_copula_pdf(F1[i-1], F2[i-1], rho), 1e-20)
        ll -= np.log(c) + np.log(f1[i-1]) + np.log(f2[i-1])
    return ll

# Define joint PDF of standard bivariate normal distribution
def bivariate_normal_pdf(x1, x2, rho):
    return (1 / (2 * np.pi * np.sqrt(1 - rho ** 2))) * np.exp(
        -1 / (2 * (1 - rho ** 2)) * (x1 ** 2 - 2 * rho * x1 * x2 + x2 ** 2)
    )

# Define inverse CDF (quantile function) of standard normal distribution
def inv_norm_cdf(u):
    return norm.ppf(u)

# Define PDF of bivariate Gaussian copula
def bivariate_gaussian_copula_pdf(u1, u2, rho):
    x1 = inv_norm_cdf(u1)
    x2 = inv_norm_cdf(u2)
    return bivariate_normal_pdf(x1, x2, rho) / (norm.pdf(x1) * norm.pdf(x2))

# Define the optimization routine
def optimize_rho(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_rho, bounds):
    # Minimize negative log-likelihood to find optimal rho
    result = min(gaussian_copula_log_likelihood, initial_rho, args=(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq),
                 bounds=[bounds])

    # Optimal value of rho
    optimal_rho = result.x[0]

    # Minimum log-likelihood value
    min_log_likelihood = result.fun

    return optimal_rho, min_log_likelihood
