import pandas as pd
import numpy as np
from scipy.stats import chi2

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

        df = pd.concat([pd.Series([p,
                                   1 - chi2.cdf(uc, 1),
                                   1 - chi2.cdf(ind, 1),
                                   1 - chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    df.columns = [var_name]
    df.index = ["EFV", "Unconditional (uc)", "Independence (ind)", "Conditional (cc)"]

    return df.round(3)


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

    df = classic_var_95.merge(garch_var_95, left_index=True, right_index=True)
    # Réinitialiser les indices de MSM_var_95 pour correspondre aux 50 derniers indices de df
    MSM_var_95.index = df.index[-100:]

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
    df_MSM_test = df[['returns_portfolio', 'MSM_var']].tail(100)
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

    df = classic_var_99.merge(garch_var_99, left_index=True, right_index=True)

    results = []

    for var in var_columns:
        test_results = christoffersen_test(df, var, 0.01)
        results.append(test_results)

    pd.set_option('display.max_columns', None)
    all_results = pd.concat(results, axis=1)
    print(all_results)
    all_results.to_csv('Christoffersen_1.csv')
