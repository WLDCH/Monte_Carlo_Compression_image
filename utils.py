import numpy as np
from random import random
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy.stats import invgamma


class GibbsSampler:


    def __init__(self, n=5, beta=2, K=10, relation=lambda x, y: x == y + 1 or x == y - 1):
        self.n = n
        self.beta = beta
        self.K = K
        self.relation = relation
        self.X0 = np.random.randint(low=1, high=self.K + 1, size=self.n)

    def get_value(self, probability_vector):
        U = random()
        for k in range(0, self.K):
            if U >= probability_vector[k] and U < probability_vector[k + 1]:
                return (k)

    def get_conditional(self, index_cond, current_X):
        somme = [0] * (self.K + 1)
        Z = 0
        for k in range(1, self.K + 1):  # on calcule p(xi=k | x_-i)

            # Calcul de la somme
            for j in range(self.n):
                if j != index_cond and self.relation(j, index_cond):
                    somme[k] += (k == current_X[j])

        Z = sum([np.exp(self.beta * somme[k]) for k in range(1, self.K + 1)])
        probability_vector = [0] * (self.K + 1)

        for k in range(1, self.K + 1):
            probability_vector[k] = 1 / Z * np.exp(self.beta * somme[k]) + probability_vector[k - 1]
        probability_vector.append(1)
        return (self.get_value(probability_vector))


    def simulate_all(self, nb_sample=500):

        X = np.zeros((self.n, nb_sample))
        X[:, 0] = self.X0

        for i in range(1, nb_sample):
            X[:, i] = X[:, i - 1]
            for k in range(self.n):
                X[k, i] = self.get_conditional(k, X[:, i])

        return (X)

    def simulate(self, nb_sample=500):
        return (self.simulate_all(nb_sample)[:, -1])

    def plot(self, nb_sample=500, composante=0, start=0):
        plt.plot(self.simulate_all(nb_sample)[composante, start:])
        plt.xlabel('Itération')
        plt.ylabel('x{}'.format(composante))
        plt.legend('Graphe de la trace de x{}'.format(composante))

    def acf_plot(self, nb_sample=500, composante=0, start=0):
        plot_acf(self.simulate_all(nb_sample)[composante, start:])


class GibbsSampler2(GibbsSampler):

    def __init__(self, n=5, beta=2, K=10, relation=lambda x, y: x == y + 1 or x == y - 1):
        super().__init__(n=5, beta=2, K=10, relation=lambda x, y: x == y + 1 or x == y - 1)
        self.mu = np.random.normal(loc=0, scale=1, size=self.K)
        self.sigma = invgamma.rvs(a=1, scale=1 ,  size=self.K)
        self.y = np.random.randint(0,255,size=(256,256)).flatten()


    def simulate(self, nb_sample=500):
        X = np.zeros((self.n, nb_sample))
        X[:, 0] = self.X0

        M = np.zeros((self.K, nb_sample), dtype=object)
        S = np.zeros((self.K, nb_sample), dtype=object)

        for i in range(self.K):
            M[i, 0] = [self.mu[i], 0, 1]
            S[i, 0] = [self.sigma[i], 1, 1]

        for i in range(1, nb_sample):

            #Simulation des xi
            X[:, i] = X[:, i - 1]
            M[:, i] = M[:, i - 1]
            S[:, i] = S[:, i - 1]
            for k in range(self.n):
                X[k, i] = self.get_conditional(k, X[:, i])

            #Simulation des sigma_i
            for k in range(self.K):
                ind = list(np.where(X[:,i]==k))
                S[k,i][1] = S[k,i][1] +len(ind)/2
                S[k,i][2] = 1/(1/S[k,i][2] + 1/2 * sum( (self.y[j]-M[k,i][0])**2 for j in ind))
                S[k,i][0] = np.random.gamma(S[k,i][1], S[k,i][2])

            #Simulation des mu_i
            for k in range(self.K):
                ind = list(np.where(X[:,i]==k))
                acc = S[k,i][0]/M[k,i][2]
                M[k,i][1] = ((sum(self.y[i] for i in ind) + M[k,i][1]*acc))/(acc+len(ind))
                M[k,i][2] = S[k,i][0]/(M[k,i][2]+len(ind))
                M[k,i] = np.random.normal(M[k,i][1], M[k,i][2] )
        return (X[:,-1],M[:,-1], S[:,-1])
