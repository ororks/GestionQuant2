import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from arch import arch_model
from scipy.stats import chi2


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
    # Initialise with variance from the first window
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

        # Calculate VaR
        std_dev = np.sqrt(today_var)
        Z_alpha = norm.ppf(1 - confidence_level)
        df.loc[i, 'VaR_riskmetrics'] = -(mean_returns.dot(weights) + Z_alpha * std_dev)

    return df


def calculate_CCC_GARCH_VaR(df, weights, initial_window_size=1135, confidence_level=0.05):
    df['VaR_CCC_GARCH'] = np.nan  # Initialize a column to store the VaRs

    # Initialize GARCH models and fit them
    volatilities = {asset: [] for asset in ['returns_SPX', 'returns_NDX']}
    residuals = {asset: [] for asset in ['returns_SPX', 'returns_NDX']}
    garch_models = {}
    for asset in ['returns_SPX', 'returns_NDX']:
        model = arch_model(df[asset][1:initial_window_size], vol='Garch', p=1, q=1, mean='Constant')
        fitted = model.fit(disp='off')
        garch_models[asset] = fitted
        volatilities[asset].extend(fitted.conditional_volatility.tolist())
        residuals[asset].extend((fitted.resid / fitted.conditional_volatility).tolist())

    # Rolling forecast using the CCC-GARCH approach
    for t in range(initial_window_size-1, len(df)):
        for asset in ['returns_SPX', 'returns_NDX']:
            model = arch_model(df[asset][1:t + 1], vol='Garch', p=1, q=1, mean='Constant')
            res = model.fit(last_obs=t + 1, disp='off')
            last_vol = res.conditional_volatility.iloc[-1]
            last_res = res.resid.iloc[-1] / last_vol

            volatilities[asset].append(last_vol)
            residuals[asset].append(last_res)

        # Update correlation matrix
        current_residuals = pd.DataFrame({asset: res for asset, res in residuals.items()})
        current_correlation = current_residuals.iloc[-initial_window_size:].corr().values

        # Construct D_t and Omega_t for the current time
        D_t = np.diag([volatilities[asset][t] for asset in ['returns_SPX', 'returns_NDX']])
        Omega_t = D_t @ current_correlation @ D_t

        # Calculate portfolio variance
        portfolio_variance = weights @ Omega_t @ weights.T

        # Calculate mean portfolio return (assuming returns are zero-centered for simplification)
        mean_portfolio_return = 0

        # Calculate VaR and store in DataFrame
        Z_alpha = norm.ppf(1 - confidence_level)
        df.loc[t+1, 'VaR_CCC_GARCH'] = -(mean_portfolio_return + np.sqrt(portfolio_variance) * Z_alpha)

    return df

def christoffersen_test(returns, var, alpha):
    """Likelihood ratio framework of Christoffersen (1998)"""
    # Calculer les violations de la VaR (rendements < -VaR de t-1)
    hits = (returns < var.shift(1))*1
    hits = hits.to_numpy()

    tr = hits[1:] - hits[:-1]  # Sequence to find transitions

    # Transitions: nij denotes state i is followed by state j nij times
    n01, n10 = (tr == 1).sum(), (tr == -1).sum()
    n11, n00 = (hits[1:][tr == 0] == 1).sum(), (hits[1:][tr == 0] == 0).sum()

    # Times in the states
    n0, n1 = n01 + n00, n10 + n11
    n = n0 + n1

    # Probabilities of the transitions from one state to another
    p01, p11 = n01 / (n00 + n01), n11 / (n11 + n10)
    p = n1 / n

    if n1 > 0:
        # Unconditional Coverage
        uc_h0 = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
        uc_h1 = n0 * np.log(1 - p) + n1 * np.log(p)
        uc = -2 * (uc_h0 - uc_h1)

        # Independence
        ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
        ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
        if p11 > 0:
            ind_h1 += n11 * np.log(p11)
        ind = -2 * (ind_h0 - ind_h1)

        # Conditional coverage
        cc = uc + ind

        # Stack results
        df = pd.concat([pd.Series(["", uc, ind, cc]),
                        pd.Series([p,
                                   1 - chi2.cdf(uc, 1),
                                   1 - chi2.cdf(ind, 1),
                                   1 - chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    # Assign names
    df.columns = ["Statistic", "Résultat/p-value"]
    df.index = ["EFV", "Unconditional (uc)", "Independence (ind)", "Conditional (cc)"]

    return df.round(3)

# =============================================================================
# Exécution du code principal
# =============================================================================
if __name__ == "__main__":
    # Extraction des données
    df = pd.read_excel(r'../datas/SP500NASDAQ.xlsx')

    # Assurez-vous que les données sont numériques
    df["SPX"] = pd.to_numeric(df["SPX"], errors="coerce")
    df["NDX"] = pd.to_numeric(df["NDX"], errors="coerce")

    # Supprimez les valeurs non numériques et les valeurs manquantes
    df = df[df["SPX"].notna() & df["NDX"].notna()]

    # Définir les poids du portefeuille
    weights = np.array([0.5, 0.5])

    # Assurez-vous que les données sont numériques
    df["SPX"] = pd.to_numeric(df["SPX"], errors="coerce")
    df["NDX"] = pd.to_numeric(df["NDX"], errors="coerce")

    # Supprimez les valeurs non numériques et les valeurs manquantes
    df = df[df["SPX"].notna() & df["NDX"].notna()]

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

        # =============================================================================
        # Création du graphique
        # =============================================================================

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df['returns_portfolio'].iloc[1135:], label='Portfolio Returns', color='blue')
        ax.plot(df['VaR_historique'].dropna(), color='red', linestyle='-', label=f'Daily VaR {100-alpha*100}% (Historical)')
        ax.plot(df['VaR_riskmetrics'].dropna(), color='lightgreen', linestyle='-', label=f'Daily VaR {100-alpha*100}% (RiskMetrics)')
        ax.plot(df['Var_cov'].dropna(), color='skyblue', linestyle='-', label=f'Daily VaR {100-alpha*100}% (Variance-Covariance)')
        ax.plot(df['VaR_CCC_GARCH'].dropna(), color='fuchsia', linestyle='-', label=f'Daily VaR {100-alpha*100}% (CCC-GARCH)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Returns / VaR')
        ax.set_title('Portfolio Returns and Daily Historical VaR Over Time')
        ax.legend(loc='upper right')
        ax.set_xlim(df.index[1135], df.index[-1])

        plt.show()

        # =============================================================================
        # Test de Christoffersen
        # =============================================================================

        # Sélectionner les rendements et les valeurs VaR pour le test
        returns = df.iloc[1135:-1]

        for var in ['VaR_historique', 'VaR_riskmetrics', 'Var_cov', 'VaR_CCC_GARCH']:
            print(f'\nVaR: {var}')
            # Effectuer le test
            test_results = christoffersen_test(returns['returns_portfolio'], returns[var], alpha)
            print(test_results)
