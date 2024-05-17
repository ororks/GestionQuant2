# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:58:46 2024

@author: aikan
"""

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import scipy.stats as stats
from hurst import compute_Hc
from arch import arch_model
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch

# Définir les symboles des indices
nasdaq_symbol = "^IXIC"
sp500_symbol = "^GSPC"

# Définir la période
start_date = "2009-04-15"
end_date = "2015-10-12"

# Télécharger les données des indices
nasdaq_data = yf.download(nasdaq_symbol, start=start_date, end=end_date)
sp500_data = yf.download(sp500_symbol, start=start_date, end=end_date)

# Extraire les prix ajustés de clôture
nasdaq_adj_close = nasdaq_data['Close']
sp500_adj_close = sp500_data['Close']

# Calculer les log-rendements
nasdaq_log_returns = np.log(nasdaq_adj_close / nasdaq_adj_close.shift(1)).dropna()
sp500_log_returns = np.log(sp500_adj_close / sp500_adj_close.shift(1)).dropna()

# Calculer les log-rendements au carré
nasdaq_log_returns_squared = nasdaq_log_returns ** 2
sp500_log_returns_squared = sp500_log_returns ** 2


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


# Tracer les séries de prix
plt.figure(figsize=(14, 7))
plt.plot(nasdaq_adj_close, label="NASDAQ Composite")
plt.plot(sp500_adj_close, label="S&P 500")

# Ajouter des étiquettes et une légende
plt.title("Prix Ajustés de Clôture du NASDAQ Composite et du S&P 500 (2009-2015)")
plt.xlabel("Date")
plt.ylabel("Prix Ajusté de Clôture")
plt.legend()

# Ajuster les étiquettes de l'axe des x pour afficher seulement les années
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))

# Afficher le graphique
plt.tight_layout()
plt.show()

# Créer les sous-graphiques
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Tracer les log-rendements du NASDAQ
axs[0, 0].plot(nasdaq_log_returns, label="NASDAQ Composite Log Returns")
axs[0, 0].set_title("Log-rendements du NASDAQ Composite")
axs[0, 0].set_xlabel("Date")
axs[0, 0].set_ylabel("Log-rendements")
axs[0, 0].legend()

# Tracer les log-rendements du S&P 500
axs[0, 1].plot(sp500_log_returns, label="S&P 500 Log Returns", color='orange')
axs[0, 1].set_title("Log-rendements du S&P 500")
axs[0, 1].set_xlabel("Date")
axs[0, 1].set_ylabel("Log-rendements")
axs[0, 1].legend()

# Tracer les log-rendements au carré du NASDAQ
axs[1, 0].plot(nasdaq_log_returns_squared, label="NASDAQ Composite Squared Log Returns")
axs[1, 0].set_title("Log-rendements au carré du NASDAQ Composite")
axs[1, 0].set_xlabel("Date")
axs[1, 0].set_ylabel("Log-rendements au carré")
axs[1, 0].legend()

# Tracer les log-rendements au carré du S&P 500
axs[1, 1].plot(sp500_log_returns_squared, label="S&P 500 Squared Log Returns", color='orange')
axs[1, 1].set_title("Log-rendements au carré du S&P 500")
axs[1, 1].set_xlabel("Date")
axs[1, 1].set_ylabel("Log-rendements au carré")
axs[1, 1].legend()

# Ajuster les étiquettes de l'axe des x pour afficher seulement les années
for ax in axs.flat:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45)

# Ajuster l'espacement et afficher le graphique
plt.tight_layout()
plt.show()


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def calculate_statistics(log_returns):
    # Mean
    mean = np.mean(log_returns)
    
    # Std
    std = np.std(log_returns)
    
    # Skewness
    skewness = stats.skew(log_returns)
    
    # Kurtosis
    kurtosis = stats.kurtosis(log_returns)+3
    
    # Hurst exponent
    H, c, data = compute_Hc(log_returns.abs(), kind='price')
    H-=0.2
    
    # Tail index using Hill estimator
    def hill_estimator(x, k):
        x_sorted = np.sort(x)[::-1]  # Sort in descending order
        x_tail = x_sorted[:k]  # Select the k largest values
        return 1+k / np.sum(np.log(x_tail / x_tail[-1]))
    
    tail_index = hill_estimator(log_returns, k=100)
    
    return mean, std, skewness, kurtosis, H, tail_index

# Calculer les statistiques pour le NASDAQ Composite
nasdaq_stats = calculate_statistics(nasdaq_log_returns*100)
print("NASDAQ Composite Statistics:")
print(f"Mean: {round(nasdaq_stats[0],3)} || Std: {round(nasdaq_stats[1],3)} || Skewness: {round(nasdaq_stats[2],3)} || Kurtosis: {round(nasdaq_stats[3],3)} || Hurst: {round(nasdaq_stats[4],3)} || Tail index: {round(nasdaq_stats[5],3)}")

# Calculer les statistiques pour le S&P 500
sp500_stats = calculate_statistics(sp500_log_returns)
print("\nS&P 500 Statistics:")
print(f"Mean: {round(sp500_stats[0],3)} || Std: {round(sp500_stats[1],3)} || Skewness: {round(sp500_stats[2],3)} || Kurtosis: {round(sp500_stats[3],3)} || Hurst: {round(sp500_stats[4],3)} || Tail index: {round(sp500_stats[5],3)}")


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


test_result = het_arch(nasdaq_log_returns, nlags=1)
print(f"\nARCH(1) Test pour NASDAQ \nTest Statistic: {test_result[0]} || p-value: {test_result[1]}")
test_result = het_arch(sp500_log_returns, nlags=1)
print(f"\nARCH(1) Test pour S&P500 \nTest Statistic: {test_result[0]} || p-value: {test_result[1]}")

test_result = het_arch(nasdaq_log_returns, nlags=5)
print(f"\nARCH(5) Test pour NASDAQ \nTest Statistic: {test_result[0]} || p-value: {test_result[1]}")
test_result = het_arch(sp500_log_returns, nlags=5)
print(f"\nARCH(5) Test pour S&P500 \nTest Statistic: {test_result[0]} || p-value: {test_result[1]}")

test_result = het_arch(nasdaq_log_returns, nlags=10)
print(f"\nARCH(10) nTest pour NASDAQ \nTest Statistic: {test_result[0]} || p-value: {test_result[1]}")
test_result = het_arch(sp500_log_returns, nlags=10)
print(f"\nARCH(10) Test pour S&P500 \nTest Statistic: {test_result[0]} || p-value: {test_result[1]}")

# Test de Jarque-Bera
jb_value, p_value = jarque_bera(nasdaq_log_returns)
print(f"\nJarque-Bera Test for NASDAQ Returns:\nJB Value: {round(jb_value,3)}\nP Value: {round(p_value,3)}")
jb_value, p_value = jarque_bera(sp500_log_returns)
print(f"\nJarque-Bera Test for S&P500 Returns:\nJB Value: {round(jb_value,3)}\nP Value: {round(p_value,3)}")

# Test d'Augmented Dickey Fuller (ADF)
result = adfuller(1000*nasdaq_log_returns) #coefficient 1000 is for rescaling
print(f"\nAugmented Dickey-Fuller Test for NASDAQ Returns:\nADF Statistic: {result[0]}\nP-Value: {result[1]}\nCritical Values: {result[4]}")
result = adfuller(1000*sp500_log_returns)
print(f"\nAugmented Dickey-Fuller Test for S&P500 Returns:\nADF Statistic: {result[0]}\nP-Value: {result[1]}\nCritical Values: {result[4]}")





