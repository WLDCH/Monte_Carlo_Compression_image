import numpy as np
from random import random
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt


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