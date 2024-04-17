import numpy as np
import pandas as pd
from numba import jit
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt



class Log_likelihood_opti(Problem):
    """
    Classe propre au package pymoo pour l'optimisation.
    On créé un classe hérite de la classe Problem du package avec
    les arguments fixes comme le nbr de composants de vol k_compos et les données.
    On initialise également :
        - n_var : le nombre de params du vecteur d'optimisation
        - n_obj : le nombre d'objectif
        - xl et xu : les limites basses et hautes des params optimisés
    """
    def __init__(self, **kwargs):
        self.k_compos = kwargs.get("k_compos")
        self.data = kwargs.get("data")
        super().__init__(n_var=4,
                         n_obj=1,
                         xl=[1.001, 1e-3, 1e-4, 1],
                         xu=[50, 0.999999, 5, 1.999999],
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Méthode héritée de la classe parent Problem propre au package.
        Dans notre cas on boucle sur plusieurs valeurs initiales du vecteurs de params
        Fonction objectif : objectif_LL
        :param x: les valeurs initiales possibles pour le vecteur de params
        :param out: le vecteur de log-vraisemblance après opti
        """
        F = np.zeros(x.shape[0])
        data = self.data["data"]
        for i in range(x.shape[0]):
            F[i] = objectif_LL(self.k_compos, data, x[i, :])
        out["F"] = F


def main_opti(data, k_compos):
    """
    Fonction principale du modèle
    :param data: Données sous forme d'un array en deux dimensions
    :param k_compos: nombre de composants de volatilités du modèle
    :return: le vecteur de volatilité estimé sur la période par le modèle
    """
    n_individuals = 20

    # Définition des limites utilisées seulement pour calcul des valeurs de l'espace de recherche
    xl = np.array([1.001, 1e-3, 1e-4, 1])
    xu = np.array([50, 0.999999, 5, 1.999999])

    # Espace de recherche des paramètres
    search_space = np.random.rand(n_individuals, len(xl)) * (xu - xl) + xl

    # Instantiation du Particle Swarm Algorithm et du problème
    algorithm = PSO(pop_size=20, sampling=search_space, adaptative=True, w=1)
    problem = Log_likelihood_opti(k_compos=k_compos, data=data)

    # Minimisation de l'inverse de la LL pour trouver le vecteur de paramètre optimal
    result = minimize(problem=problem,
                      algorithm=algorithm,
                      seed=1,
                      verbose=True)
    params_opti = result.X

    # Prédiction du vecteur de vol avec le vecteur de params opti
    likelihood = estimate_vol(params_opti, k_compos, data)
    likelihood["params"] = params_opti
    likelihood["n"] = 252
    pred = msm_predict(likelihood['g_m'], likelihood['params'][2], likelihood['n'],
                       likelihood['filtered'], likelihood['A'], h=None)

    return pred

def compute_wt(data, s):
    pa = (2 * np.pi) ** (-0.5)
    w_t = data
    w_t = pa * np.exp(-0.5 * ((w_t / s) ** 2)) / s
    # On ajoute une faible variation pour éviter les div par 0
    w_t = w_t + 1e-16
    return w_t

def objectif_LL(k_compos, data, theta):
    """return LL, the vector of log likelihoods
    """

    # Initialisation et récupérations des paramètres et constantes
    b = theta[0]
    gamma_k = theta[1]
    sigma = theta[2]
    m0 = theta[3]
    k_compos2 = 2 ** k_compos
    T = len(data)

    # Matrice de probas de transitions
    A = compute_transition_matrix(k_compos, b, gamma_k)

    # Valeurs possibles du vecteur de composants de vol
    g_m = compute_states_vector(k_compos, m0)

    # Valeurs possible de la vol d'après le processus supposé par le modèle
    s = sigma * g_m

    # Matrice omega des probas P(rendements|M_t=mi) pour tous i dans les 2^k valeurs possibles du vecteur de vol
    # et toutes périodes t
    w_t = compute_wt(data, s)

    # log likelihood using numba
    LL,_,_ = compute_loglikelihood(k_compos2, T, A, w_t)

    return (LL)


@jit(nopython=True)
def compute_loglikelihood(k_compos2, T, A, w_t):
    """
    Fonction de calcul de la log-likelihood
    """
    # Initialisation du vecteur des log-likelihood des observations
    LLs = np.zeros(T)
    # matrice pour contenir les vecteur pi_t càd pour chaque date le vecteur qui contient
    # les probas que le vec M = état j conditionellement aux rendements pour tous j dans les k^2 possibilités
    # := P(M_t = m_j|rendements)
    pi_mat = np.zeros((T + 1, k_compos2))
    # Initialisation du vecteur pi_0 avec les probas ergodiques càd juste 1/k^2
    pi_mat[0, :] = (1 / k_compos2) * np.ones(k_compos2)

    for t in range(T):
        # On multiplie pour chaque période t le vec 1*k^2 de probas P(M_t=m_j|rendements) avec la matrice de transition
        # ce qui revient à intégrer P(M_t=m_j|rendements)P(M_t+1=m_j|M_t=i) à travers tout |M_t=i) et pour tous j
        # Donc piA = [P(M_t = m1), ..., P(M_t=md)]
        piA = np.dot(pi_mat[t, :], A)
        # C = [P(rendements|M_t = m1)*P(M_t=m1),....., P(rendements|M_t=md)*P(M_t=md)]
        C = (w_t[t, :] * piA)
        # on intégre sur tout j=1,...,d dans M_t = mj donc on obtient P(rendements)
        ft = np.sum(C)

        if abs(ft - 0) <= 1e-05: # Permet d'éviter les divs par zéro
            pi_mat[t + 1, 1] = 1
        else:
            # Règle de Bayes classique pour calculer le nouveau vecteur de prob
            # P(A|B) = P(B|A)*P(A)/P(B) <=> P(M_t+1 = m_j|rendements) = P(rendements|M_t = m1)*P(M_t=m1)/P(rendements)
            pi_mat[t + 1, :] = C / ft

        # Vecteur de lls
        LLs[t] = np.log(np.dot(w_t[t, :], piA))

    LL = -np.sum(LLs)

    return LL, LLs, pi_mat


def compute_transition_matrix(k_compos, b, gamma_k):
    """
    Fonction de calcul des probas de transition d'état gamma.
    Etape 1 : On calcule les proba gamma et leurs complémentaires dans une matrice k*2
            Ces probas inconditionelles représentent la probabilité de changer d'état
            ou de rester dans le même état pour les k composants du vecteur de volatilité.
            -> matric de taille k*2
    Etape 2 : On calcule les probas conditionnelles comme le produit entre toutes les combinaisons
            de probas possibles -> vecteur de taille 2^k
    Etape 3 : On créé la matrice composé du vecteur prob créé à l'étape 2 -> matrice de taille 2^k*2^k
    """

    # compute gammas
    gamma = np.zeros((k_compos, 1))
    # On initialise la première valeur de gamma en isolant gamma_1 à partir de la formule des auteurs
    gamma[0, 0] = 1 - (1 - gamma_k) ** (1 / (b ** (k_compos - 1)))
    # On calcule les k-1 probas gamma suivantes en colonne
    for i in range(1, k_compos):
        gamma[i, 0] = 1 - (1 - gamma[0, 0]) ** (b ** (i-1))
    # Pas compris d'où ça sort et c'est pas une erreur car aussi dans le code des auteurs
    gamma = gamma * 0.5
    # On concatène pour avoir deux colonnes de probs
    gamma = np.c_[gamma, gamma]
    gamma[:, 0] = 1 - gamma[:, 0]

    # Probas de transitions du vecteur M de composants de vol
    k_compos2 = 2 ** k_compos
    prob = np.ones(k_compos2)

    # On itère pour les k^2 valeurs possibles du vecteur de composants de vol
    for i in range(k_compos2):
        for m in range(k_compos):
            # On créé un array d'1 élément en représentation 16-bit qu'on découpe en 2 8-bit
            # Exemple i=1 => tmp = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            tmp = np.unpackbits(
                np.arange(i, i + 1, dtype=np.uint16).view(np.uint8))
            # On échange les 8 premiers et 8 derniers éléments de place
            # Donc : i=1 => tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            tmp = np.append(tmp[8:], tmp[:8])
            # Pour chaque valeur possible du vec M, on obtient toute les probas de transitions
            # en faisant le produit de toutes les combinaisons de probas inconditionelles possible.
            # Exemple : Pour k=2, i=1 et start m=0 : 1*gamma[2-0-1, tmp[-(0+1)]] => 1*gamma[1, 1]
            # Puis pareil pour m=1 : 1*gamma[1, 1]*gamma[0, 0] => proba du vec M avec 2 composants de vol : 1 qui reste
            # dans le même état (1-gamma) et un 2nd qui change d'état (gamma).
            prob[i] = prob[i] * gamma[k_compos - m - 1, tmp[-(m + 1)]]
    # On place ces probas dans la matrice de transition avec np.fromfunction qui créé une matrice dont les valeurs pour
    # chaque coordonnée i,j sont données par la valeur du vecteur prob correspondant à l'indice donné par
    # une opé xor entre les versions 8-bit des indices i et j.
    # Exemple pour k_comp=2 , i=1, j = 2 : xor = [0, 0, 0, 0, 0, 0, 1, 1] = 3 donc prob[3] en A(1,2)
    A = np.fromfunction(
        lambda i, j: prob[np.bitwise_xor(i, j)], (k_compos2, k_compos2), dtype=np.uint16)

    return (A)



@jit(nopython=True)
def compute_states_vector(k_compos, m0):
    """
    Méthode de calcul de toutes les valeurs possibles du vecteur d'état M
    Rappel : le vecteur d'état M en t contient k composants en t.
    La vol dépend du produit des K composants
    A un instand donné chaque composant du vecteur M peut prendre la valeur m1 ou m0.
    Donc le produit des composants du vecteur M peut prendre 2^k valeurs possibles
    return: vecteur des 2^k valeurs possibles
    """
    # Initialisation des valeurs possibles de la distribution binomiale
    m1 = 2 - m0
    k_compos2 = 2 ** k_compos
    # Initialisation du vecteur qui va contenir les 2^k valeurs possibles
    state_values = np.zeros(k_compos2)
    # Initialisation d'un vecteur qui contient toutes les valeurs de 0 à 2^k-1
    sv_range = np.arange(k_compos2)
    # Pour chaque valeurs de 0 à 2^k-1
    for i in range(k_compos2):
        # Initialisation de la valeur pour le premier état = 1
        state = 1
        for j in range(k_compos):
            # On compare la représentation binaire du ième élément de sv_range
            # avec la représentation binaire de 2**j. On check chaque bit avec la règle du AND
            # Si il y a un bit en commun, le résultat est =! de 0.
            # Cela permet de parcourir toutes les 2^k combinaisons possibles du produit des k composants de vol.
            if np.bitwise_and(sv_range[i], (2 ** j)) != 0:
                state = state * m1
            else:
                state = state * m0
        state_values[i] = state

    return (np.sqrt(state_values))


def msm_predict(g_m, sigma, n, pi_mat, A, h=None):
    """
    Fonction de calcul du vecteur de vol estimé
    """
    # Check du steps h pour le forecast
    if h is not None and h < 1:
        raise ValueError("h must be a non-zero integer")
    if h is not None:
        h = int(h)

    sigma = sigma #/ np.sqrt(n)

    if h is not None:
        p_hat = np.dot(pi_mat[-1, :].reshape(1, -1), np.linalg.matrix_power(A, h))
        vol = sigma * np.dot(p_hat, g_m)
    else:

        vol = sigma * np.dot(pi_mat, g_m)

    return vol


def estimate_vol(para, k_compos, data, n_vol=252):
    """
    Version modifiée de objectif_LL utilisée en dehors de l'optimisation
    pour renvoyer aussi la matrice de probas de transition, la matrice pmat
    des probas de M_t à chaque période, et g_m les valeurs possibles de M
    """
    # Initialisation et récupérations des paramètres et constantes
    b = para[0]
    gamma_k = para[1]
    sigma = para[2]
    m0 = para[3]
    k_compos2 = 2 ** k_compos
    T = len(data)

    # Matrice de probas de transitions
    A = compute_transition_matrix(k_compos, b, gamma_k)

    # Valeurs possibles du vecteur de composants de vol
    g_m = compute_states_vector(k_compos, m0)

    # Valeurs possible de la vol d'après le processus supposé par le modèle
    s = sigma * g_m

    # Matrice omega des probas P(rendements|M_t=mi) pour tous i dans les 2^k valeurs possibles du vecteur de vol
    # et toutes périodes t
    w_t = compute_wt(data, s)

    LL, LLs, pmat = compute_loglikelihood(k_compos2, T, A, w_t)
    likelihood = {'LL': LL}
    likelihood['filtered'] = pmat[1:, :]
    likelihood['A'] = A
    likelihood['g_m'] = g_m

    return likelihood



if __name__ == "__main__":
    # Extraction des données
    datas = pd.read_csv(r'C:\Users\roman\OneDrive\Bureau\M2 EIF\S2\Projet_GQ2\datas\SP500NASDAQ.csv')

    df = pd.DataFrame(datas)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)

    df["SP500"] = pd.to_numeric(df["SP500"], errors="coerce")
    df = df[df["SP500"] != "."]

    data_test = df["SP500"].to_numpy(dtype=np.float64)
    data_test = (np.log(data_test[1:]) - np.log(data_test[0:-1]))

    # Suppresion des valeurs NaN
    data_test = data_test[~np.isnan(data_test)]

    # On centre les résidus comme dans l'article
    data_test = data_test - data_test.mean()

    # ajout d'une nouvelle dim en colonne
    # donc devient un vecteur d'array de taille T*1
    data_test = data_test[:, np.newaxis]

    # Nombre de composants de volatilité
    k_compos = 7 #7 comme dans notre article
    data = data_test

    # Appel de l'algo pour estimer la vol
    result = main_opti(data, k_compos)
    print(result)

    valeurs_plot = pd.DataFrame()
    valeurs_plot["volatilité daily estimée"] = result
    valeurs_plot["volatilité annuelle estimée"] = result*np.sqrt(252)
    valeurs_plot["carré des rendements centrés"] = data_test**2
    valeurs_plot["carré des rendements centrés * np.sqrt(252)"] = (data_test**2)*np.sqrt(252)

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    for i, col in enumerate(valeurs_plot.columns):
        axs[i].plot(valeurs_plot.index, valeurs_plot[col], label=col)
        axs[i].set_title(f"{col}")
        axs[i].set_xlabel("Temps")
        axs[i].set_ylabel("Estimateur")
        axs[i].legend()
    plt.tight_layout()

    plt.show()

