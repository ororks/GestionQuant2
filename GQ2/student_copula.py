import numpy as np
from scipy.stats import t
from scipy.optimize import minimize as min

def bivariate_student_pdf(x1, x2, rho, df):
    # Gamma function from numpy
    gamma = np.math.gamma
    
    print("root : ", 1 - rho ** 2)
    print("df : ", df)
    # Part of the denominator
    denom = np.sqrt(1 - rho ** 2) * np.pi * df * gamma(df / 2)

    # The exponent part
    exponent = -((df + 2) / 2)

    # The z term in the distribution formula
    z = (x1 ** 2 - 2 * rho * x1 * x2 + x2 ** 2) / (1 - rho ** 2)

    # Complete PDF calculation
    pdf = (gamma((df + 2) / 2) / denom) * (1 + z / df) ** exponent

    return pdf

def inv_stu_cdf(u, df):
    return t.ppf(u, df)

def bivariate_student_copula_pdf(u1, u2, rho, df):
    x1 = inv_stu_cdf(u1, df)
    x2 = inv_stu_cdf(u2, df)
    return bivariate_student_pdf(x1, x2, rho, df) / (t.pdf(x1, df) * t.pdf(x2, df))


def student_copula_log_likelihood(param, f1, f2, F1, F2):
    rho = param[0]
    df = param[1]
    ll = 0
    for i in range(len(f1)+1):
        c = max(bivariate_student_copula_pdf(F1[i-1], F2[i-1], rho, df), 1e-20)
        ll -= np.log(c) + np.log(f1[i-1]) + np.log(f2[i-1])
    return ll

def optimize_rho_df(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq, initial_rho_df, bounds):
    # Minimize negative log-likelihood to find optimal rho
    result = min(student_copula_log_likelihood, initial_rho_df, args=(fy_sp500, fy_nasdaq, Fy_sp500, Fy_nasdaq),
                 bounds=[bounds])

    # Optimal value of rho
    optimal_rho = result.x[0]

    # Minimum log-likelihood value
    min_log_likelihood = result.fun

    return optimal_rho, min_log_likelihood