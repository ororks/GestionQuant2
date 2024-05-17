import matplotlib.pyplot as plt
import pandas as pd

def plot_var_95(df, alpha):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['returns_portfolio'], label='Portfolio Returns', color='blue')
    ax.plot(df['VaR_historique'].dropna(), color='red', linestyle='-', label=f'Historical')
    ax.plot(df['VaR_riskmetrics'].dropna(), color='lightgreen', linestyle='-', label=f'RiskMetrics')
    ax.plot(df['Var_cov'].dropna(), color='skyblue', linestyle='-', label=f'Variance-Covariance')
    ax.plot(df['VaR_CCC_GARCH'].dropna(), color='fuchsia', linestyle='-', label=f'CCC-GARCH')
    ax.plot(df['Student'].dropna(), color='yellow', linestyle='-', label=f'Student-Copula-GARCH')
    ax.plot(df['MSM_var'].dropna(), color='black', linestyle='-', label=f'Student-Copula-MSM')

    ax.set_xlabel('Time')
    ax.set_ylabel('VaR (%)')
    ax.set_title(f'VaR {100-alpha}% Over Time')
    ax.legend(loc='upper right')
    ax.set_xlim(df.index[0], df.index[-1])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0))

    plt.savefig(rf'../VaR_{100-alpha}_plot.png')

    plt.show()

def plot_var_99(df, alpha):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['returns_portfolio'], label='Portfolio Returns', color='blue')
    ax.plot(df['VaR_historique'].dropna(), color='red', linestyle='-', label=f'Historical')
    ax.plot(df['VaR_riskmetrics'].dropna(), color='lightgreen', linestyle='-', label=f'RiskMetrics')
    ax.plot(df['Var_cov'].dropna(), color='skyblue', linestyle='-', label=f'Variance-Covariance')
    ax.plot(df['VaR_CCC_GARCH'].dropna(), color='fuchsia', linestyle='-', label=f'CCC-GARCH')
    ax.plot(df['Student'].dropna(), color='yellow', linestyle='-', label=f'Student-Copula-GARCH')

    ax.set_xlabel('Time')
    ax.set_ylabel('VaR (%)')
    ax.set_title(f'VaR {100-alpha}% Over Time')
    ax.legend(loc='upper right')
    ax.set_xlim(df.index[0], df.index[-1])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0))

    plt.savefig(rf'../VaR_{100-alpha}_plot.png')

    plt.show()

def plot_var_garch(df, alpha):
    fig, ax = plt.subplots(figsize=(12, 6))

    for column in df.columns:
        if column != 'returns_portfolio':
            ax.plot(df[column].dropna(), label=column)

    ax.plot(df['returns_portfolio'], label='Portfolio Returns', color='blue')

    ax.set_xlabel('Time')
    ax.set_ylabel('VaR (%)')
    ax.set_title(f'VaR {100-alpha}% Over Time')
    ax.legend(loc='upper right')
    ax.set_xlim(df.index[0], df.index[-1])
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0))

    plt.savefig(rf'../VaR_Garch_{100-alpha}_plot.png')

    plt.show()


if __name__ == "__main__":

    classic_var_95 = pd.read_excel(rf'../datas/VaR_0.05.xlsx')
    classic_var_95 = classic_var_95[['Dates', 'returns_portfolio', 'VaR_historique', 'VaR_riskmetrics', 'Var_cov', 'VaR_CCC_GARCH']]
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

    plot_var_95(df, 5)

    # =============================================================================
    # Plot Var 99
    # =============================================================================

    classic_var_99 = pd.read_excel(rf'../datas/VaR_0.01.xlsx')
    classic_var_99 = classic_var_99[['Dates', 'returns_portfolio', 'VaR_historique', 'VaR_riskmetrics', 'Var_cov', 'VaR_CCC_GARCH']]
    classic_var_99 = classic_var_99.iloc[-500:]

    garch_var_99 = pd.read_csv(rf'../datas/Garch_VaR_99.csv')

    # Joindre les données
    df = classic_var_99.merge(garch_var_99, left_index=True, right_index=True)

    plot_var_99(df, 1)

    # =============================================================================
    # Plot les Garch Var
    # =============================================================================

    returns_portfolio = df['returns_portfolio']

    garch_var_95 = garch_var_95.join(returns_portfolio)
    plot_var_garch(garch_var_95, 5)

    garch_var_99 = garch_var_99.join(returns_portfolio)
    plot_var_garch(garch_var_99, 1)
