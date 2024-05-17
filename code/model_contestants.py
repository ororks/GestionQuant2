import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.special import expit

def christoffersen_test(df, var_name, alpha):
    """Likelihood ratio framework of Christoffersen (1998)"""

    returns = df['returns_portfolio']
    var = df[var_name]

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
        df = pd.concat([pd.Series([p,
                                   1 - chi2.cdf(uc, 1),
                                   1 - chi2.cdf(ind, 1),
                                   1 - chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    # Assign names
    df.columns = [var_name]
    df.index = ["EFV", "Unconditional (uc)", "Independence (ind)", "Conditional (cc)"]

    return df.round(3)


# Fonction pour calculer la perte basée sur VaR
def calculate_var_loss(df, var_column, alpha):
    loss = (alpha - (df['returns_portfolio'] < df[var_column]).astype(int)) * (df['returns_portfolio'] - df[var_column])
    return loss


# Fonction de bootstrap stationnaire
def stationary_bootstrap(data, B=5000, w=0.1):
    n = len(data)
    resamples = np.empty((B, n))
    for b in range(B):
        resample = np.empty(n)
        i = np.random.randint(n)
        for t in range(n):
            if np.random.rand() < w:
                i = np.random.randint(n)
            resample[t] = data[i]
            i = (i + 1) % n
        resamples[b, :] = resample
    return resamples


# Fonction pour effectuer le test SPA
def spa_test(df, var_columns, alpha, base_var):
    T = len(df)
    losses = {var: calculate_var_loss(df, var, alpha).dropna().values for var in var_columns}
    loss_means = {var: np.mean(loss) for var, loss in losses.items()}

    RP = {var: losses[base_var] - loss for var, loss in losses.items()}
    RP_mean = {var: np.mean(rp) for var, rp in RP.items()}
    RP_var = {var: np.var(rp, ddof=1) for var, rp in RP.items()}

    T_SPA = max([np.sqrt(T) * RP_mean[var] / np.sqrt(RP_var[var]) for var in RP if var != base_var])

    # Générer des échantillons bootstrap pour calculer les p-values
    B = 5000
    bootstrap_results = {}
    RP_base = losses[base_var]
    for var in RP:
        if var != base_var:
            RP_diff = RP_base - losses[var]
            bootstrap_samples = stationary_bootstrap(RP_diff, B=B)
            RP_bootstrap_mean = np.mean(bootstrap_samples, axis=1)
            RP_bootstrap_var = np.var(bootstrap_samples, axis=1, ddof=1)
            T_SPA_bootstrap = [np.sqrt(T) * mean / np.sqrt(var) for mean, var in
                               zip(RP_bootstrap_mean, RP_bootstrap_var)]
            p_value = np.mean(np.array(T_SPA_bootstrap) >= T_SPA)
            bootstrap_results[var] = p_value

    return bootstrap_results

# =============================================================================
# Exécution du code principal
# =============================================================================
if __name__ == "__main__":

    classic_var_95 = pd.read_excel(rf'../datas/VaR_0.05.xlsx')
    classic_var_95 = classic_var_95[
        ['Dates', 'returns_portfolio', 'VaR_historique', 'VaR_riskmetrics', 'Var_cov', 'VaR_CCC_GARCH']]
    classic_var_95 = classic_var_95.iloc[-500:]

    garch_var_95 = pd.read_csv(rf'../datas/Garch_VaR_95.csv')

    MSM_var_95 = pd.read_csv(rf'../datas/MSM_VaR.csv')
    MSM_var_95 = MSM_var_95.rename(columns={'0': 'MSM_var'})
    MSM_var_95['MSM_var'] = MSM_var_95['MSM_var'].apply(lambda x: x * 100)

    # Joindre les données
    df = classic_var_95.merge(garch_var_95, left_index=True, right_index=True)
    # Réinitialiser les indices de MSM_var_95 pour correspondre aux 50 derniers indices de df
    MSM_var_95.index = df.index[-50:]
    # Joindre df et MSM_var_95
    df = df.join(MSM_var_95)

    # =============================================================================
    # Test de Christoffersen
    # =============================================================================

    var_columns = df.drop(['Dates', 'MSM_var', 'returns_portfolio'], axis=1).columns.tolist()

    results = []

    for var in var_columns:
        test_results = christoffersen_test(df, var, 0.05)
        results.append(test_results)

    # test MSM 95%
    df_MSM_test = df[['returns_portfolio', 'MSM_var']].tail(50)
    test_results_MSM = christoffersen_test(df_MSM_test, 'MSM_var', 0.05)
    results.append(test_results_MSM)

    pd.set_option('display.max_columns', None)
    all_results = pd.concat(results, axis=1)
    print(all_results)
    all_results.to_csv('Christoffersen_5.csv')


    ### 99% VaR
    classic_var_99 = pd.read_excel(rf'../datas/VaR_0.01.xlsx')
    classic_var_99 = classic_var_99[
        ['Dates', 'returns_portfolio', 'VaR_historique', 'VaR_riskmetrics', 'Var_cov', 'VaR_CCC_GARCH']]
    classic_var_99 = classic_var_99.iloc[-500:]

    garch_var_99 = pd.read_csv(rf'../datas/Garch_VaR_99.csv')

    # Joindre les données
    df = classic_var_99.merge(garch_var_99, left_index=True, right_index=True)

    results = []

    for var in var_columns:
        # Effectuer le test
        test_results = christoffersen_test(df, var, 0.01)
        results.append(test_results)

    pd.set_option('display.max_columns', None)
    all_results = pd.concat(results, axis=1)
    print(all_results)
    all_results.to_csv('Christoffersen_1.csv')


    # =============================================================================
    # Test de Hansen
    # =============================================================================

    '''base_var = 'VaR_historique'  # Exemple de modèle de référence
    spa_results = spa_test(df, var_columns, alpha, base_var)

    print(f"\nRésultats du test SPA à {100 * (1 - alpha)}% par rapport à {base_var} :")
    for var, p_value in spa_results.items():
        print(f"{var}: p-value = {p_value:.4f}")'''
