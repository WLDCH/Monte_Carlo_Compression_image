import numpy as np
from random import random
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from scipy.stats import invgamma
from PIL import Image, ImageOps
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')



    
def show(img):
    plt.figure()
    plt.imshow(img.astype(int), cmap='gray', vmin=0, vmax=255)





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



    
class ImageCompression():
    
    def __init__(self, y=np.random.randint(0,255,size=(256,256)), K=16,  beta=1):
       
        self.y = y
        self.u, self.v = self.y.shape
        self.K = K
        self.x = np.round((self.y/255)*self.K, 0).astype(int)
        self.beta = beta
        self.m_k = np.arange(0, 255, 255/self.K)
        self.s_k = [1] * self.K
        self.alpha_k = np.array([1] * self.K)
        self.beta_k = np.array([10] * self.K)
        self.mu = np.random.normal(self.m_k, self.s_k, self.K)
        self.sigma = np.random.gamma(self.alpha_k, 1/self.beta_k, self.K)
        
    @staticmethod    
    def find_neighbors(i,j,x,y):
        if i==0:
            if j==0:
                return [(1,0),(0,1)]
            elif j==y-1:
                return [(1,y-1),(0,y-2)]
            else:
                return [(0,j-1),(0,j+1),(1,j)]
        elif i==x-1:
            if j==0:
                return [(i-1,j),(i,j+1)]
            elif j==y-1:
                return [(i-1,j),(i,j-1)]
            else:
                return [(i,j-1),(i,j+1),(i-1,j)]
        else:
            if j==0:
                return [(i+1,j),(i,j+1), (i-1,j)]
            elif j==y-1:
                return [(i-1,j),(i,j-1), (i+1,j)]
            else:
                return [(i,j-1),(i,j+1),(i-1,j), (i+1,j)]

    @staticmethod
    def f_normal(y, mu, sigma_2):
        return 1 / sigma_2 ** 0.5 * np.exp(-(y-mu)**2/(2*sigma_2))  

    @staticmethod
    def choice(distrib):
        distrib = distrib.cumsum()
        t = np.random.rand()
        i = 0
        while i < len(distrib) and distrib[i] < t:
            i += 1
        return i + 1
    
    @staticmethod
    def convert(path, resize=128):
        img = Image.open(path)
        img.thumbnail((resize, resize), Image.ANTIALIAS)  
        gray_image = ImageOps.grayscale(img)
        return(np.array(gray_image))
    
    
    
    def gibbs_sampling(self, nb_iter=10): 

    
        for _ in tqdm(range(nb_iter)):
            
            
            # Partie sur x
            for i in range(self.u):
                for j in range(self.v):
                    self.x[i,j] = -1
                    l_neighbors = ImageCompression.find_neighbors(i,j,self.u,self.v)
                    distrib = np.array([0]*self.K)
                    for neigh in l_neighbors:
                        distrib[self.x[neigh]-1] += 1
                    #print(distrib)
                    distrib = np.exp(self.beta * (distrib-distrib[0])) # On fait -distrib[0] afin de ne pas avoir une exponentielle trop grande
                    for k in range(self.K):
                        distrib[k] *= ImageCompression.f_normal(self.y[i,j], self.mu[k], self.sigma[k])
                    #print(distrib)
                    #print('__________________________')
                    distrib = distrib / np.sum(distrib)
                    try:
                        self.x[i,j] = ImageCompression.choice(distrib)
                    except ValueError:
                        print(i,j,distrib)
            
            
            # Partie sur mu
            for k in range(self.K):
                n_k = 0
                buf_k = 0
                for i in range(self.u):
                    for j in range(self.v):
                        if self.x[i, j] == k:
                            n_k += 1
                            buf_k += self.y[i, j] / self.sigma[k-1]
                var_mu = (self.sigma[k-1]**0.5 * self.s_k[k-1])**2/(self.sigma[k-1] + self.s_k[k-1]**2 *n_k)
                f_k = self.m_k[k-1]/self.s_k[k-1]**2 + buf_k
                self.mu[k-1] = np.random.normal(var_mu * f_k, np.sqrt(var_mu)) 

            
            # Partie sur sigma
            for k in range(self.K):
                n_k = 0
                buf_k = 0
                for i in range(self.u):
                    for j in range(self.v):
                        if self.x[i, j] == k:
                            n_k += 1
                            buf_k += (self.y[i, j] - self.mu[k-1])**2 / 2
                self.sigma[k-1] = np.random.gamma(n_k/2 + self.alpha_k[k-1], 1/(self.beta_k[k-1] + buf_k)) 

        return(self.x, self.mu, self.sigma)
        
    
    def compress(self, nb_iter=10):
        mat = np.zeros((self.u,self.v))
        X, _, _ = self.gibbs_sampling()
        for i in range(self.u):
            for j in range(self.v):
                mat[i,j] = self.mu[X[i,j]-1]
        return mat
    
