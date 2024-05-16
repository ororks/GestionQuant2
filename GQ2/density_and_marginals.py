import numpy as np
import itertools
from scipy.stats import norm

def calcualte_density(y, pmat, sigma, m0, k_compos):
    """
    calcule la densité conditionnelle de y conditionnellement aux états du modèle et à leurs probabilités
    """
    # 1/vol selon les états du modèle
    # permet de calculer la densité de y/vol => eps selon les états
    denum = calculate_denum(m0, sigma, k_compos)
    # initialisation du vecteur
    density_f = np.zeros(len(y))

    for j in range(len(y)+1):
        density_f[j-1] = calc_density_t(y[j - 1], denum, pmat[j - 1])

    return density_f


def calc_density_t(y, denum, prob):
    """
    calcule la densité comme somme pondérée de probabilité de l'état et densité de eps dans l'état
    """
    # calcule eps en fonction de y et de la vol dans l'état
    eps = denum * y
    # densité de eps
    density = norm.pdf(eps)
    # densité conditionnelle
    cond_density = density * denum

    return np.dot(cond_density, prob)

def calculate_denum(m0, sigma, k_compos):
    # Define your values
    values = [m0, 2-m0]

    # Generate all possible combinations of the product of the three values
    combinations = list(itertools.product(values, repeat=k_compos))

    # Calculate the product for each combination
    products = [np.prod(comb) for comb in combinations]

    return [(1 / (sigma * np.sqrt(val))) for val in products]

def calcualte_marginals(y, pmat, sigma, m0, k_compos):

    denum = calculate_denum(m0, sigma, k_compos)
    marginal_f = np.zeros(len(y))

    for j in range(len(y)+1):

        marginal_f[j - 1] = calc_marginal_t(y[j - 1], denum, pmat[j - 1])

    return marginal_f

def calc_marginal_t(y, denum, prob):

    cdff = norm.cdf(denum * y)

    return np.dot(cdff, prob)
