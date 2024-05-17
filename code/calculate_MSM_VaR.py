import density_and_marginals
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import root


class Calculate_VaR:
    """
    Calcule la VaR à partir des densités conditionnelles et de la densité de copule
    """
    def __init__(self, pmat_1, sigma_1, m0_1, pmat_2, sigma_2, m0_2, k_compos, copula_params, copula_density_function, alpha, tolerence_threshold):
        self.pmat_1 = pmat_1
        self.sigma_1 = sigma_1
        self.m0_1 = m0_1
        self.pmat_2 = pmat_2
        self.sigma_2 = sigma_2
        self.m0_2 = m0_2
        self.k_compos = k_compos
        self.copula_params = copula_params
        self.copula_density_function = copula_density_function
        self.alpha = alpha
        self.tolerence_threshold = tolerence_threshold


    def VaR_calculation(self):
        """
        Création de la boucle et restitution des param de la vol stochastique selon le modèle MSM
        """

        denum1 = density_and_marginals.calculate_denum(self.m0_1, self.sigma_1, self.k_compos)
        denum2 = density_and_marginals.calculate_denum(self.m0_2, self.sigma_2, self.k_compos)
        VaR = np.zeros(len(self.pmat_1))

        for j in range(len(self.pmat_1) + 1):
            print("j : ", j+1)

            VaR[j - 1] = self.opti_VaR_stu(denum1, denum2, self.pmat_1[j - 1], self.pmat_2[j - 1], self.alpha)

        return VaR

    def opti_VaR_stu(self, denum1, denum2, prob1, prob2, alpha):
        x0 = np.array([-0.025])
        result = root(self.VaR_resolution, x0, args=(denum1, denum2, prob1, prob2, alpha))
        return result.x[0]

    def VaR_resolution(self, z, denum1, denum2, prob1, prob2, alpha):
        x0, x1 = -0.1, z  # Limits for x
        y0, y1 = -0.1, 0.1  # Limits for y (constants in this case)
        result, error = dblquad(self.joined_density, x0, x1, lambda x: y0, lambda x: y1,
                                args=(denum1, denum2, prob1, prob2))
        return result - alpha

    def joined_density(self, u, v, denum1, denum2, prob1, prob2):
        F1 = density_and_marginals.calc_marginal_t(np.array([u]), denum1, prob1)
        F2 = density_and_marginals.calc_marginal_t(np.array([v]), denum2, prob2)
        result = self.copula_density_function(F1, F2, self.copula_params)
        result = (result * density_and_marginals.calc_density_t(np.array([u]), denum1, prob1)
                  * density_and_marginals.calc_density_t(np.array([v]), denum2, prob2))
        return result

