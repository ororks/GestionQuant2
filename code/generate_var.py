import pandas as pd
import numpy as np
from scipy.stats import norm
from arch import arch_model

def historical_VaR(df, confidence_level, window_size=1135):
    df['VaR_historique'] = np.nan
    for i in range(window_size, len(df)):
        df.loc[df.index[i], 'VaR_historique'] = np.percentile(df['returns_portfolio'].iloc[i-window_size:i].dropna(),
                                                              confidence_level)

    return df

def variance_covariance_method(df, weights, window_size=1135, confidence_level = 0.05):
    df['Var_cov'] = np.nan

    for i in range(window_size, len(df)):
        # Sélection de la fenêtre pour les calculs
        window = df.iloc[i - window_size:i]
        mean_returns = window[['returns_SPX', 'returns_NDX']].mean()
        cov_matrix = window[['returns_SPX', 'returns_NDX']].cov()

        # Calcul de la variance du portefeuille
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Moyenne des rendements du portefeuille
        mean_portfolio_return = np.dot(weights, mean_returns)

        # Quantile de la distribution normale
        Z_alpha = norm.ppf(1 - confidence_level)

        # Calcul de la VaR
        VaR = -(mean_portfolio_return + portfolio_std * Z_alpha)
        df.loc[df.index[i], 'Var_cov'] = VaR

    return df


def riskmetrics_VaR(df, weights, window_size=1135, lambda_param=0.94, confidence_level=0.05):
    df['VaR_riskmetrics'] = np.nan

    # Sélection de la fenêtre pour les calculs
    initial_window = df.iloc[:window_size]
    cov_matrix = initial_window[['returns_SPX', 'returns_NDX']].cov()
    initial_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    df.loc[window_size-1, 'portfolio_variance'] = initial_variance
    for i in range(window_size, len(df)):
        yesterday_var = df.loc[i - 1, 'portfolio_variance']
        today_return = df.loc[i, 'returns_portfolio']
        today_var = lambda_param * yesterday_var + (1 - lambda_param) * today_return ** 2
        df.loc[i, 'portfolio_variance'] = today_var
        mean_returns = df[['returns_SPX', 'returns_NDX']].iloc[i-window_size:i].mean()

        # Calcul de la VaR
        std_dev = np.sqrt(today_var)
        Z_alpha = norm.ppf(1 - confidence_level)
        df.loc[i, 'VaR_riskmetrics'] = -(mean_returns.dot(weights) + Z_alpha * std_dev)

    return df


def calculate_CCC_GARCH_VaR(df, weights, initial_window_size=1135, confidence_level=0.05):
    df['VaR_CCC_GARCH'] = np.nan

    # Initialisation du modèle GARCH et fit
    volatilities = {asset: [] for asset in ['returns_SPX', 'returns_NDX']}
    residuals = {asset: [] for asset in ['returns_SPX', 'returns_NDX']}
    garch_models = {}
    for asset in ['returns_SPX', 'returns_NDX']:
        model = arch_model(df[asset][1:initial_window_size], vol='Garch', p=1, q=1, mean='Constant')
        fitted = model.fit(disp='off')
        garch_models[asset] = fitted
        volatilities[asset].extend(fitted.conditional_volatility.tolist())
        residuals[asset].extend((fitted.resid / fitted.conditional_volatility).tolist())

    # Rolling forecast du CCC-GARCH
    for t in range(initial_window_size-1, len(df)):
        for asset in ['returns_SPX', 'returns_NDX']:
            model = arch_model(df[asset][1:t + 1], vol='Garch', p=1, q=1, mean='Constant')
            res = model.fit(last_obs=t + 1, disp='off')
            last_vol = res.conditional_volatility.iloc[-1]
            last_res = res.resid.iloc[-1] / last_vol

            volatilities[asset].append(last_vol)
            residuals[asset].append(last_res)

        # MaJ de la correlation matrix
        current_residuals = pd.DataFrame({asset: res for asset, res in residuals.items()})
        current_correlation = current_residuals.iloc[-initial_window_size:].corr().values

        # Construction de  D_t et Omega_t au temps t
        D_t = np.diag([volatilities[asset][t] for asset in ['returns_SPX', 'returns_NDX']])
        Omega_t = D_t @ current_correlation @ D_t

        # Calculate portfolio variance
        portfolio_variance = weights @ Omega_t @ weights.T

        # On assume que les rendements sont nuls
        mean_portfolio_return = 0

        # Calcul de la VaR
        Z_alpha = norm.ppf(1 - confidence_level)
        df.loc[t+1, 'VaR_CCC_GARCH'] = -(mean_portfolio_return + np.sqrt(portfolio_variance) * Z_alpha)

    return df


# =============================================================================
# Exécution du code principal
# =============================================================================
if __name__ == "__main__":
    # Extraction des données
    df = pd.read_excel(r'../datas/SP500NASDAQ.xlsx')
    df["SPX"] = pd.to_numeric(df["SPX"], errors="coerce")
    df["NDX"] = pd.to_numeric(df["NDX"], errors="coerce")
    df = df[df["SPX"].notna() & df["NDX"].notna()]

    # Définir les poids du portefeuille
    weights = np.array([0.5, 0.5])

    # Calculez les rendements log des données
    df['returns_SPX'] = np.log(df['SPX']).diff() * 100
    df['returns_NDX'] = np.log(df['NDX']).diff() * 100

    # Pondérez les rendements en fonction de leur poids dans le portefeuille
    df['returns_portfolio'] = weights[0] * df['returns_SPX'] + weights[1] * df['returns_NDX']

    df_ini = df.copy()

    for alpha in [0.05, 0.01]:
        print(f'\nVaR à  {100-alpha*100}%')
        df = df_ini.copy()

        # =============================================================================
        # Utilisation des modèles VaR
        # =============================================================================

        var_historique = historical_VaR(df, confidence_level=alpha * 100)
        var_cov = variance_covariance_method(df, weights, confidence_level=alpha)
        var_riskmetrics = riskmetrics_VaR(df, weights, confidence_level=alpha)
        var_ccc_garch = calculate_CCC_GARCH_VaR(df, weights, confidence_level=alpha)

        returns = df.iloc[1135:-1]
        returns.to_excel(f'../datas/VaR_{alpha}.xlsx')

