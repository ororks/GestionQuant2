
from scipy.optimize import minimize
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib import callbacks
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
pandas2ri.activate()
import os


def rpy2_output_error(output):
    try:
        decoded_output = output.decode('utf-8')
        print(decoded_output)
    except UnicodeDecodeError as e:
        print("R output contained characters that could not be decoded:", e)

callbacks.consolewrite_print = rpy2_output_error
callbacks.consolewrite_warnerror = rpy2_output_error

class Garch():
    def __init__(self, returns, price, theta, copula_garch: bool = False, returns2=None, price2=None, copula_type='Normal', h=None):

        self.returns = np.array(returns)
        self.price = price
        self.theta = np.array(theta)
        self.copula_type = copula_type
        self.setup_parameters(self.returns, h)
        self.h = h
        if copula_garch:
            self.returns2 = np.array(returns2)
            self.price2 = price2
            self.setup_parameters(self.returns2, self.h, suffix='2')
        self.copula_garch = copula_garch


    def setup_parameters(self, returns, h, suffix=''):
        setattr(self, f'var_incondi{suffix}', np.var(returns))
        theta_opti, _ = self.optim(returns, getattr(self, f'var_incondi{suffix}'))
        setattr(self, f'theta_opti{suffix}', theta_opti)
        var_condi = self.fit(theta_opti, returns, getattr(self, f'var_incondi{suffix}'))
        setattr(self, f'var_condi{suffix}', var_condi)
        residuals = self.compute_residuals(theta_opti, returns, var_condi)
        setattr(self, f'residuals{suffix}', residuals)
        predicted_variance = self.predict_variance(returns,
                                                   theta_opti,
                                                   var_condi,
                                                   getattr(self, f'var_incondi{suffix}'),
                                                   h)
        setattr(self, f'predicted_variance{suffix}', predicted_variance)
        predicted_mean = self.predict_mean(theta_opti, h)
        setattr(self, f'predicted_mean{suffix}', predicted_mean)

    def get_predicted_mean(self, suffix=''):
        return getattr(self, f'predicted_mean{suffix}')

    def get_var_condi(self, suffix=''):
        return getattr(self, f'var_condi{suffix}')

    def get_var_incondi(self, suffix=''):
        return getattr(self, f'var_incondi{suffix}')

    def get_theta_opti(self, suffix=''):
        return getattr(self, f'theta_opti{suffix}')

    def get_predicted_variance(self, suffix=''):
        return getattr(self, f'predicted_variance{suffix}')

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

    def predict_mean(self, theta_opti, h):
        mu, _, _ = theta_opti
        predicted_mean = [mu] * h
        return predicted_mean

    def predict_variance(self, returns, theta_opti, var_condi, var_incondi, h):
        mu, alpha, beta = theta_opti
        w = (1 - alpha - beta) * var_incondi
        var_last = w + alpha*((returns[-1]-mu)**2) + beta*var_condi[-1]
        variance_predictions = [var_last]
        for h in range(1, h):
            variance_prediction = (w * np.sum([(alpha+beta)**i for i in range(h-1)]) +
                                   (alpha+beta)**(h-1) * (alpha*((returns[-1]-mu)**2)+beta*var_condi[-1]))
            variance_predictions.append(variance_prediction)
        return variance_predictions

    def save_residuals(self, file_name='residuals.csv'):
        if self.copula_garch and hasattr(self, 'residuals2'):

            if self.residuals is not None and self.residuals2 is not None:
                df = pd.DataFrame({
                    'Residuals1': self.residuals,
                    'Residuals2': self.residuals2
                })

                directory = "C:/Users/roman/OneDrive/Bureau/M2EIF/S2/Projet_GQ2/code"
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

    def fit_copula_garch(self, residuals_1, residuals_2):
        script_path = "C:/Users/roman/OneDrive/Bureau/M2EIF/S2/Projet_GQ2/code/fit_copula.R"

        robjects.r(f'source("{script_path}")')

        df = pd.DataFrame({
            'Residuals1': residuals_1,
            'Residuals2': residuals_2
        })
        r_dataframe = pandas2ri.py2rpy(df)

        try:
            r_fit_copula = robjects.globalenv['fit_copula']
            result = r_fit_copula(r_dataframe, self.copula_type)
            simulated_residuals = np.array(result)
            return simulated_residuals
        except Exception as e:
            print("Error during copula fitting:", e)
            return None


    def fitted_copula_params(self):
        window_size = 1135
        returns_window = self.returns
        returns2_window = self.returns2

        theta_opti_1, _ = self.optim(returns_window, self.get_var_incondi())
        var_condi_1 = self.fit(theta_opti_1, returns_window, self.get_var_incondi())
        residuals_1 = self.compute_residuals(theta_opti_1, returns_window, var_condi_1)
        theta_opti_2, _ = self.optim(returns2_window, self.get_var_incondi(suffix='2'))
        var_condi_2 = self.fit(theta_opti_2, returns2_window, self.get_var_incondi(suffix='2'))
        residuals_2 = self.compute_residuals(theta_opti_2, returns2_window, var_condi_2)

        script_path = "C:/Users/roman/OneDrive/Bureau/M2EIF/S2/Projet_GQ2/code/fit_copula.R"

        robjects.r(f'source("{script_path}")')

        df = pd.DataFrame({
            'Residuals1': residuals_1,
            'Residuals2': residuals_2
        })
        r_dataframe = pandas2ri.py2rpy(df)

        try:
            r_fit_copula_params = robjects.globalenv['fit_copula_params']
            result = r_fit_copula_params(r_dataframe, self.copula_type)
            print(result)
        except Exception as e:
            print("Error during copula fitting:", e)
            return None



    def main_loop(self):
        T = len(self.returns)
        window_size = 1135
        perc95_VaR = []
        perc99_VaR = []
        print("Start estimation of the VaR")
        for i in tqdm(range(500)):
            print(f'iteration {i}/500')
            start_index = i
            end_index = window_size + i
            returns_window = self.returns[start_index:end_index]
            returns2_window = self.returns2[start_index:end_index]

            theta_opti_1, _ = self.optim(returns_window, self.get_var_incondi())
            var_condi_1 = self.fit(theta_opti_1, returns_window, self.get_var_incondi())
            residuals_1 = self.compute_residuals(theta_opti_1, returns_window, var_condi_1)
            var_condi_1_pred = self.predict_variance(returns_window, theta_opti_1, var_condi_1, self.get_var_incondi(), h=1)
            mean_1_pred = self.predict_mean(theta_opti_1, 1)
            theta_opti_2, _ = self.optim(returns2_window, self.get_var_incondi(suffix='2'))
            var_condi_2 = self.fit(theta_opti_2, returns2_window, self.get_var_incondi(suffix='2'))
            residuals_2 = self.compute_residuals(theta_opti_2, returns2_window, var_condi_2)
            var_condi_2_pred = self.predict_variance(returns2_window, theta_opti_2, var_condi_2,
                                                     self.get_var_incondi(suffix='2'), h=1)
            mean_2_pred = self.predict_mean(theta_opti_2, 1)
            MC_residuals = self.fit_copula_garch(residuals_1, residuals_2)

            portfolio = []
            for j in range(len(MC_residuals)):
                simu_resid_1 = MC_residuals[j, 0]
                simu_resid_2 = MC_residuals[j, 1]
                simu_return_1 = np.sqrt(var_condi_1_pred) * simu_resid_1 + mean_1_pred
                simu_return_2 = np.sqrt(var_condi_2_pred) * simu_resid_2 + mean_2_pred
                price_1_t_1 = self.price[end_index - 1]
                price_2_t_1 = self.price2[end_index - 1]
                portfolio_value = ((np.exp(simu_return_1)*price_1_t_1 - price_1_t_1) + (
                            np.exp(simu_return_2)*price_2_t_1 - price_2_t_1))/2
                portfolio.append(portfolio_value/10)

            portfolio.sort()
            perc95_VaR.append(np.percentile(portfolio, 5))
            perc99_VaR.append(np.percentile(portfolio, 1))

        return perc95_VaR, perc99_VaR

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
    df = pd.read_excel('C:/Users/roman/OneDrive/Bureau/M2EIF/S2/Projet_GQ2/datas/SP500NASDAQ2.xls', header=0,
                       index_col=0)
    NASDAQ_price = df.iloc[:, 0]
    SP500_price = df.iloc[:, 1]
    NASDAQ_logreturn = np.log(df['NASDAQCOM']).diff().dropna()
    SP500_logreturn = np.log(df['SP500']).diff().dropna()
    mu = np.mean(SP500_logreturn)
    initial_params = [mu, 0.1, 0.1]
    model = Garch(returns=SP500_logreturn,
                  price=SP500_price,
                  theta=initial_params,
                  copula_garch=True,
                  returns2=NASDAQ_logreturn,
                  price2=NASDAQ_price,
                  copula_type="Plackett",
                  h=1)

    perc95_VaR, perc99_VaR = model.main_loop()
    print(f"95% VaR: {perc95_VaR}")
    print(f"99% VaR: {perc99_VaR}")

    plt.figure(figsize=(10, 6))
    plt.plot(perc95_VaR, label='95% VaR', color='blue')
    plt.plot(perc99_VaR, label='99% VaR', color='red')
    plt.xlabel('Time Period')
    plt.ylabel('VaR Value')
    plt.title('Value at Risk (VaR) over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
