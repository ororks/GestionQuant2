import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import os

class Garch():
    def __init__(self, returns, theta, copula_garch: bool = False, returns2=None, copula_type='Normal'):

        self.returns = np.array(returns)
        self.theta = np.array(theta)
        self.copula_type = copula_type
        self.setup_parameters(self.returns)
        if copula_garch:
            self.returns2 = np.array(returns2)
            self.setup_parameters(self.returns2, suffix='2')
        self.copula_garch = copula_garch
        self.results_copula = self.fit_copula_garch()

    def setup_parameters(self, returns, suffix=''):
        setattr(self, f'var_incondi{suffix}', np.var(returns))
        theta_opti, _ = self.optim(returns, getattr(self, f'var_incondi{suffix}'))
        setattr(self, f'theta_opti{suffix}', theta_opti)
        var_condi = self.fit(theta_opti, returns, getattr(self, f'var_incondi{suffix}'))
        setattr(self, f'var_condi{suffix}', var_condi)
        residuals = self.compute_residuals(theta_opti, returns, var_condi)
        setattr(self, f'residuals{suffix}', residuals)

    def model(self, theta, returns, var_incondi):
        mu, alpha, beta = theta
        w = (1 - alpha - beta) * var_incondi
        e = np.concatenate([[0], (returns - mu)**2])
        T = len(e)
        var_condi = np.full(T, var_incondi)
        for i in range(1, T):
            var_condi[i] = w + alpha * e[i-1] + beta * var_condi[i-1]
        var_condi[var_condi <= 0] = np.min(var_condi[var_condi > 0])
        ll = 0.5 * np.log(2 * np.pi) + 0.5 * np.log(var_condi[1:]) + 0.5 * (e[1:] / var_condi[1:])
        return np.sum(ll)

    def optim(self, returns, var_incondi):
        bounds = [(1e-8, None), (1e-8, 0.9999), (1e-8, 0.9999)]
        result = minimize(self.model, self.theta, args=(returns, var_incondi), bounds=bounds, tol=1e-8)
        return result.x, result.fun

    def fit(self, theta_opti, returns, var_incondi):
        mu, alpha, beta = theta_opti
        w = (1 - alpha - beta) * var_incondi
        e = np.concatenate([[0], (returns - mu) ** 2])
        T = len(e)
        var_condi = np.full(T, var_incondi)
        for i in range(1, T):
            var_condi[i] = w + alpha * e[i - 1] + beta * var_condi[i - 1]
        return var_condi

    def compute_residuals(self, theta_opti, returns, var_condi):
        mu, _, _ = theta_opti
        std_dev = np.sqrt(var_condi)
        return (returns - mu) / std_dev[1:]

    def save_residuals(self, file_name='residuals.csv'):
        if self.copula_garch and hasattr(self, 'residuals2'):

            if self.residuals is not None and self.residuals2 is not None:
                df = pd.DataFrame({
                    'Residuals1': self.residuals,
                    'Residuals2': self.residuals2
                })

                directory = "C:/Users/roman/OneDrive/Bureau/M2 EIF/S2/Projet_GQ2/code"
                full_path = os.path.join(directory, file_name)

                if not os.path.exists(directory):
                    os.makedirs(directory)

                try:
                    df.to_csv(full_path, index=False)
                    print(f"Residues standardisés enregistrés à {full_path}")
                except Exception as e:
                    print(f"Erreur dans le save: {e}")
            else:
                print("Problème avec les données de résidus.")
        else:
            print("Résidus pas dispos pour les deux séries")

    def fit_copula_garch(self):
        self.save_residuals()
        if os.path.exists('residuals.csv'):
            try:
                robjects.r.assign("copula_type", self.copula_type)

                robjects.r.source('fit_copula.R')

                if 'fit_results' in robjects.globalenv:
                    fit_results = robjects.globalenv['fit_results']
                    print(fit_results if isinstance(fit_results, str) else fit_results.r_repr())
                else:
                    print("Résultats pas trouvés dans l'env R")
            except Exception as e:
                print(f"{str(e)}")
        else:
            print("Fichier des résidus pas trouvé")

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.var_condi, label='Conditional Variance', color='blue', linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Variance')
        ax.set_title('GARCH Model Conditional Variance Over Time')
        ax.legend(loc='upper right')

        ax2 = ax.twinx()
        ax2.plot(self.returns, label='Returns', color='green')
        ax2.set_ylabel('Returns')
        ax2.legend(loc='upper left')

        plt.show()

if __name__ == "__main__":
    datas = pd.read_csv(r'C:\Users\roman\OneDrive\Bureau\M2 EIF\S2\Projet_GQ2\datas\SP500NASDAQ.csv')

    df = pd.DataFrame(datas)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")
    df = df[df["SP500"] != "."]
    df = df.dropna()
    SP500_returns = np.log(df['SP500']).diff().dropna() * 100

    mu = np.mean(SP500_returns)
    initial_params = [mu, 0.1, 0.1]
    df["NASDAQCOM"] = pd.to_numeric(df["NASDAQCOM"], errors="coerce")
    df = df[df["NASDAQCOM"] != "."]
    df = df.dropna()
    data_test = df["NASDAQCOM"].to_numpy(dtype=np.float64)
    NASDAQ_returns = np.log(df['NASDAQCOM']).diff().dropna() * 100
    print(NASDAQ_returns)
    model = Garch(returns=SP500_returns,
                  theta=initial_params,
                  copula_garch=True,
                  returns2=NASDAQ_returns,
                  copula_type="rotGumbel")
    print(model.results_copula)
