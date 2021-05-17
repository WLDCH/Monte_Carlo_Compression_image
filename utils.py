import numpy as np
from random import random, choice
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm
from PIL import Image, ImageOps
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')


def show(array):
    """

    :type array: 2D np.ndarray with int values between 0 and 255
    :return: grayscale image corresponding to the argument array
    """
    plt.figure()
    plt.imshow(array.astype(int), cmap='gray', vmin=0, vmax=255)


def inv_cdf_discret(probability_vector):
    """
    :param probability_vector: vector representing the PDF of a discrete distribution
    :return: a simulation of a random variable follow probability_vector distribution
    """
    u = random()
    i = 0
    while i < len(probability_vector) and probability_vector[i] < u:
        i += 1
    return i + 1


class GibbsSampler:

    def __init__(self, n=5, beta=2, K=10, relation=lambda x, y: x == y + 1 or x == y - 1):
        """
        :param n: number of components of the Potts model
        :param beta: beta parameter of the Potts model
        :param K: number of values that can be taken by the components of the Potts model
        :param relation: function used to determine the neighbors of a given pixel
        """
        self.n = n
        self.beta = beta
        self.k_potts = K
        self.relation = relation
        self.potts_values = np.random.randint(low=1, high=self.k_potts + 1, size=self.n)

    def get_conditional(self, index_cond, current_X):
        """
        :param index_cond: index of the component that we want to update
        :param current_X: current value of the Potts model
        :return: the updated value of that component
        """
        somme = [0] * (self.k_potts + 1)
        probability_vector = np.array([0] * (self.k_potts + 1))
        for k in range(1, self.k_potts + 1):
            # Computing p(xi=k | x_-i) for every k
            for j in range(self.n):
                if j != index_cond and self.relation(j, index_cond):
                    somme[k] += (k == current_X[j])
            probability_vector[k] = np.exp(self.beta * somme[k])
        probability_vector = probability_vector / np.sum(probability_vector)
        return inv_cdf_discret(np.cumsum(probability_vector))

    def simulate_all(self, nb_iter=500):
        """
        Iterates the Gibbs sampler
        :return: the evolution of the Potts model
        """
        potts_evol = np.zeros((self.n, nb_iter))
        potts_evol[:, 0] = self.potts_values

        for i in range(1, nb_iter):
            potts_evol[:, i] = potts_evol[:, i - 1]
            for k in range(self.n):
                potts_evol[k, i] = self.get_conditional(k, potts_evol[:, i])

        return potts_evol

    def simulate(self, nb_sample=500):
        return self.simulate_all(nb_sample)[:, -1]

    def plot(self, nb_sample=500, composante=0, start=0):
        plt.plot(self.simulate_all(nb_sample)[composante, start:])
        plt.xlabel('Itération')
        plt.ylabel('x{}'.format(composante))
        plt.legend('Graphe de la trace de x{}'.format(composante))
        plt.show()

    def acf_plot(self, nb_sample=500, composante=0, start=0):
        plot_acf(self.simulate_all(nb_sample)[composante, start:])


class ImageCompression:

    def __init__(self, path=None, image_size=128, K_potts=10, beta_potts=1, mu_mean=1, mu_std=1, alpha=1, beta=10):
        """
        :param path: Path of the image we want to compress. If no path is selected, a random gray scale image is
                        generated
        :param image_size: the passed image will be resized to this size
        :param K_potts: number of values possible for each component of the Potts model
        :param beta_potts: The Beta parameter of the Potts model
        :param mu_mean: parameter to compute the initial means of the mu vector
        :param mu_std: parameter to compute the initial std of the mu vector
        :param alpha: initial alpha parameter for the InvGamma law
        :param beta: initial beta parameter for the InvGamma law
        :var self.mu : initial value of the mu vector
        :var self.sigma: initial value of the sigma vector
        :var self.potts_values: initial value of each pixel in the Potts model
        """
        if path is None:
            self.image = np.random.randint(0, 256, size=(image_size, image_size))
        else:
            image = Image.open(path)
            image.thumbnail((image_size, image_size), Image.ANTIALIAS)
            gray_image = ImageOps.grayscale(image)
            self.image = np.array(gray_image)
        self.rows, self.columns = self.image.shape
        self.k_potts = K_potts
        self.beta_potts = beta_potts
        self.mu_mean = np.arange(0, 255, 255 * mu_mean / self.k_potts)
        self.mu_std = [mu_std] * self.k_potts
        self.alpha = np.array([alpha] * self.k_potts)
        self.beta = np.array([beta] * self.k_potts)
        self.mu = np.random.normal(self.mu_mean, self.mu_std, self.k_potts)
        self.sigma = invgamma.rvs(self.alpha, scale=self.beta, size=self.k_potts)
        self.potts_values = np.zeros((self.rows, self.columns))

    @staticmethod
    def find_neighbors(i, j, rows, columns):
        """
        :param i: abscissa of the pixel we're interested in
        :param j: ordinate of the pixel we're interested in
        :param rows: number of rows in the grid where we're searching for neighbors
        :param columns: number of columns in the grid where we're searching for neighbors
        :return: list containing the coordinates of the adjacent neighbors of the pixel at (i, j)
        """
        if i == 0:
            if j == 0:
                return [(1, 0), (0, 1)]
            elif j == columns - 1:
                return [(1, columns - 1), (0, columns - 2)]
            else:
                return [(0, j - 1), (0, j + 1), (1, j)]
        elif i == rows - 1:
            if j == 0:
                return [(i - 1, j), (i, j + 1)]
            elif j == columns - 1:
                return [(i - 1, j), (i, j - 1)]
            else:
                return [(i, j - 1), (i, j + 1), (i - 1, j)]
        else:
            if j == 0:
                return [(i + 1, j), (i, j + 1), (i - 1, j)]
            elif j == columns - 1:
                return [(i - 1, j), (i, j - 1), (i + 1, j)]
            else:
                return [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)]

    @staticmethod
    def f_normal(y, mean, sigma_2):
        """
        :return: value of the normal pdf at y
        """
        return 1 / sigma_2 ** 0.5 * np.exp(-(y - mean) ** 2 / (2 * sigma_2))

    def gibbs_sampling(self, nb_sample=500):
        """
        :param nb_sample: number of iterations of the Gibbs sampling algorithm
        :return: potts_evol: matrxi giving the evolution of the values of each pixel in the Potts model
                 mu_evol: evolution of the value of the mu vector
                 sigma_evol: evolution of the value of the sigma vector
        """
        potts_evol = np.zeros((self.rows, self.columns, nb_sample)).astype(int)
        mu_evol = np.zeros((self.k_potts, nb_sample))
        sigma_evol = np.zeros((self.k_potts, nb_sample))

        potts_evol[..., 0] = self.potts_values
        mu_evol[:, 0] = self.mu
        sigma_evol[:, 0] = self.sigma

        for time in tqdm(range(1, nb_sample)):

            potts_evol[..., time] = potts_evol[..., time - 1]
            mu_evol[:, time] = mu_evol[:, time - 1]
            sigma_evol[:, time] = sigma_evol[:, time - 1]

            # we start by updating the values of the Potts model
            for row in range(self.rows):
                for col in range(self.columns):
                    somme = [0] * self.k_potts
                    probability_vector = [0] * self.k_potts
                    for k in range(self.k_potts):
                        list_neighbors = self.find_neighbors(row, col, self.rows, self.columns)
                        for neigh in list_neighbors:
                            somme[k] += (potts_evol[..., time][neigh] == k + 1)

                        probability_vector[k] = np.exp(self.beta_potts * somme[k]) * \
                                                self.f_normal(self.image[row, col], mu_evol[k, time],
                                                              abs(sigma_evol[k, time]))
                    probability_vector = [probability / sum(probability_vector) for probability in probability_vector]
                    cdf_vector = np.cumsum(probability_vector)
                    potts_evol[row, col, time] = inv_cdf_discret(cdf_vector)

            # we now update the sigma vector given the updated Potts model
            for k in range(self.k_potts):
                # we first retrieve the indexes of the updated Potts model pixels that are equal to k+1
                rows, columns = np.where(potts_evol[..., time] == k + 1)
                n_k = len(rows)
                ind_k = [[rows[i], columns[i]] for i in range(len(rows))]
                # we then simulate the next sigma[k] thanks to its posterior distribution (given by Bayesian theory)
                alpha_k = self.alpha[k] + n_k / 2
                beta_k = self.beta[k] + 1 / 2 * sum([(self.image[index[0], index[1]] - mu_evol[k, time]) ** 2
                                                     for index in ind_k])
                sigma_evol[k, time] = invgamma.rvs(alpha_k, scale=beta_k)

            # We finally update the mu vector given the updated Potts model and sigma vector
            for k in range(self.k_potts):
                # we first retrieve the indexes of the updated Potts model pixels that are equal to k+1
                rows, columns = np.where(potts_evol[..., time] == k + 1)
                n_k = len(rows)
                ind_k = [[rows[i], columns[i]] for i in range(len(rows))]
                # we then simulate the next mu[k] thanks to its posterior distribution (given by Bayesian theory)
                var_mu = (sigma_evol[k, time] ** 0.5 * self.mu_std[k]) ** 2 / (
                        sigma_evol[k, time] + self.mu_std[k] ** 2 * n_k)
                f_k = self.mu_mean[k] / self.mu_std[k] ** 2 + sum(
                    [self.image[index[0], index[1]] / sigma_evol[k, time] for index in ind_k])
                mu_evol[k, time] = np.random.normal(var_mu * f_k, np.sqrt(var_mu))

        return potts_evol, mu_evol, sigma_evol

    def compress(self, nb_sample=100):
        """
        :return: this methods replaces each pixel in the image with the final mu[k], where k is the value of that pixel
                    in the Potts model
        """
        potts_evol, mu_evol, sigma_evol = self.gibbs_sampling(nb_sample)
        potts_model = potts_evol[..., -1]
        mu = mu_evol[:, -1]
        compressed_img = np.zeros((self.rows, self.columns))
        for i in range(self.rows):
            for j in range(self.columns):
                compressed_img[i, j] = mu[potts_model[i, j] - 1]
        return compressed_img

    def compress2(self, nb_sample=100):
        """
        :return: this method replaces each pixel in the image with final mu[k] +/- the final sigma[k], where k is the
                    value of that pixel in the Potts model
        """
        potts_evol, mu_evol, sigma_evol = self.gibbs_sampling(nb_sample)
        potts_model = potts_evol[..., -1]
        mu = mu_evol[:, -1]
        sigma = sigma_evol[:, -1]
        compressed_img = np.zeros((self.rows, self.columns))
        for i in range(self.rows):
            for j in range(self.columns):
                compressed_img[i, j] = mu[potts_model[i, j] - 1] + \
                                       choice([1, -1]) * np.sqrt(sigma[potts_model[i, j] - 1])
        return compressed_img


if __name__ == "__main__":
    img = ImageCompression('ENSAE_image.jpg', K_potts=10)
    compressed_meth1 = img.compress(nb_sample=100)
    show(compressed_meth1)
    plt.show()
