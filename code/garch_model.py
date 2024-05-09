import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Garch():
    def __init__(self, returns, theta):
        self.returns = np.array(returns)
        self.theta = np.array(theta)
        self.mu = theta[0]
        self.alpha = theta[1]
        self.beta = theta[2]
        self.var_incondi = np.var(self.returns)
        self.e = None
        self.theta_opti, _ = self.optim()
        _, self.maxLL = self.optim()
        self.var_condi = self.fit()


    def model(self, theta):
        mu, alpha, beta = theta
        w = (1 - alpha - beta) * self.var_incondi
        e = np.concatenate([[0], (self.returns - mu)**2])
        T = len(e)
        var_condi = np.full(T, self.var_incondi)

        for i in range(1, T):
            var_condi[i] = w + alpha * e[i-1] + beta*var_condi[i-1]

        var_condi[var_condi<=0] = np.min(var_condi[var_condi>0])

        ll = 0.5 * np.log(2 * np.pi) \
             + 0.5 * np.log(var_condi[1:]) \
             + 0.5 * (e[1:] / var_condi[1:])

        LL = np.sum(ll)
        return LL

    def optim(self):
        bounds = [(1e-8, None), (1e-8, 0.9999), (1e-8, 0.9999)]
        result = minimize(lambda x: self.model(x), self.theta, bounds=bounds, tol=1e-8)
        return result['x'], result['fun']

    def fit(self):
        mu, alpha, beta = self.theta_opti
        w = (1 - alpha - beta) * self.var_incondi
        e = np.concatenate([[0], (self.returns - mu) ** 2])
        T = len(e)
        var_condi = np.full(T, self.var_incondi)

        for i in range(1, T):
            var_condi[i] = w + alpha * e[i - 1] + beta * var_condi[i - 1]

        return var_condi

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

    # Fit du Garch avec mon implementation
    mu = np.mean(SP500_returns)
    initial_params = [mu, 0.1, 0.1]
    model = Garch(SP500_returns, initial_params)
    print(model.theta_opti)
    print(model.maxLL)
    model.plot()
    plt.show()
