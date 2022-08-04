# Project Simulation et Monte-Carlo

**CHEDRI Axel, CHRIMNI Walid, DUFLOT Quentin**

## Introduction

Le modèle de Potts est une généralisation du modèle d’Ising permettant un nombre de modalités supérieur à deux : On considère un vecteur aléatoire $x=(x_1,...,x_n)$ à valeurs dans $\{1, . . . , K\}^n \, K > 2$, dont la probabilité est:


$π(x) = \frac{1}{\text{Z}(β)} \text{exp}\left\lbrace β \sum_{i∼j}^{} \mathbb{1}[x_i = x_j])\right\\rbrace$


où
* $β$ > 0 
*  Z($β$) = $ \sum_{x_1,...,x_n \in \{1,...,K\}^n}^{} \text{exp}\left\{β \sum_{i∼j}^{} \mathbb{1}[x_i = x_j])\right\} $ est une constante de normalisation
* i ∼ j est une relation de voisinage (par exemple j est un des quatre pixels
adjacents au pixel i, dans le cas où les xi représentent les pixels d’une image
rectangulaire).

Ce projet est construit sur la programmation orientée objet. Toutes nos fonctions sont basées sur des classes. Les raisons principales à cela sont les suivantes :

* Utilisation et réutilisation plus facile 
* Un code plus élégant et clair
* Maintenance et évolutivité plus efficace

Toutes nos classes se trouvent dans le fichier MonteCarloClasses.py que l'on importe dans la cellule suivante. Les packages de bases dont nous avons besoin (numpy, matplotlib etc...) sont également présenté dans le module MonteCarloClasses

En outre, pour plus de clarté les codes utilisés pour construire nos classes ne seront pas montrés dans ce notebook. Cependant, vous trouverez ci-dessous un "mode d'emploi" pour savoir comment utilisé chaque classe.

### Mode d'emploi

Pour créer un objet de la classe *GibbsSampler*, il suffit de lancer la commande  ``` GibbsSampler(n, beta, K, relation) ``` où
* n correspond à la taille du vecteur $x$
* beta correspond au paramètre $β$
* K correspond au paramètre K
* relation est une fonction Python telle que relation(i,j) = 1 si i~j

Des valeurs sont affectés par défaut à ces paramètres, il est donc possible de créer une instance de classe seulement via ```GibbsSampler()```

Les méthodes ```plot(nb_sample, composante, start)``` et ``` plot_acf(nb_sample, composante, start)``` permettent de tracer respectivement le graphe au cours du temps et le graphe de correlation où
* nb_sample correspond au nombre d'itération de l'algorithme pour la simulation de l'echantillon
* compostante correspond à la compostante du vecteur que l'on veut simuler
* start correspond au temps à partir duquel on affiche le graphe (pour les problèmes où la convergence est plus lente)



Pour créer un objet de la classe *ImageCompression*, il suffit de lancer la commande  ``` ImageCompression(n, beta, K, relation) ``` où
* path : chemin de l'image choisie. Si rien n'est mentionnée ou None, alors un vecteur de variables entière aléatoires entre 1 et 255 sera généré.
* image_size: l'image entrée sera redimensionnée
* K_potts : nombre de valeur possibles pour les variables de Potts
* beta_potts : paramètre $\beta$ de Potts 
* mu_mean : paramètre de moyenne initiale des $\mu_k$ de loi normale
* mu_std paramètre d'écart-type initiale des $\mu_k$ de loi normale
* alpha: paramètre alpha initiale des $\sigma_k$ suivant une loi inverse gamma
* beta: paramètre beta initiale des $\sigma_k$ suivant une loi inverse gamma


**Pour toutes les méthodes, des valeurs par défauts sont affectées en paramètres afin de permettre de lancer rapidement un programme** (voir utils.py pour les valeur par défauts


```python
from utils import *
```

## Question 1 

> Pour β et K fixé, proposer un Gibbs sampler pour simuler selon la loi du modèle. Déterminer la performance de l’algorithme en fonction de ces paramètres.

Pour la création technique du modèle et du code associé, voir la création de la classe "GibbsSampler" du fichier "MonteCarloClasses.py"

L'échantillonnage de Gibbs est une méthode MCMC (Monte-Carlo par chaînes de Markov). Etant donné une distribution de probabilité $\pi$ sur un univers $\Omega$, cet algorithme définit une chaîne de Markov dont la distribution stationnaire est $\pi$. Il permet ainsi de tirer aléatoirement un élément de $\Omega$ selon la loi $\pi$ (on parle d'échantillonnage).

Dans un premier temps, nous créons notre instance de classe :


```python
gibbs = GibbsSampler(n=30, beta=1, K=10, relation=lambda x,y : x==y+1 or x==y-1 or x==y-2 or x==y+2 or x==y-3 or x==y+3)
```

On génère la réalisation de la variable aléatoire $x$ qui suit la loi mentionnée en introduction :


```python
gibbs.simulate(nb_sample=500)
```




    array([3., 6., 3., 3., 3., 8., 2., 6., 0., 8., 5., 4., 5., 4., 4., 3., 3.,
           1., 0., 2., 0., 3., 8., 3., 2., 2., 1., 9., 8., 0.])




```python
gibbs.simulate(nb_sample=500).shape
```




    (30,)



C'est un vecteur de taille 30 (=n), l'entrée i de l'array correspond à la réalisation de $x_i$

Visualisons la trace de $x_1$ :


```python
gibbs.plot(nb_sample=300)
```


    
![png](readme_files/readme_11_0.png)
    


La chaine "mélange" bien, la trace de $x_1$ semble bien être random. Confirmons cela par un graphe d'autocorellation :


```python
gibbs.acf_plot(nb_sample=300, composante=0, start=250)
```


    
![png](readme_files/readme_13_0.png)
    



```python
gibbs = GibbsSampler(n=30, beta=1, K=20, relation=lambda x,y : x==y+1 or x==y-1 or x==y-2 or x==y+2 or x==y-3 or x==y+3)
```


```python
gibbs.plot(nb_sample=300)
gibbs.acf_plot(nb_sample=300, composante=0, start=250)
```


    
![png](readme_files/readme_15_0.png)
    



    
![png](readme_files/readme_15_1.png)
    



```python
gibbs = GibbsSampler(n=30, beta=1, K=100, relation=lambda x,y : x==y+1 or x==y-1 or x==y-2 or x==y+2 or x==y-3 or x==y+3)
```


```python
gibbs.plot(nb_sample=300)
gibbs.acf_plot(nb_sample=300, composante=0, start=250)
```


    
![png](readme_files/readme_17_0.png)
    



    
![png](readme_files/readme_17_1.png)
    


Le graphe d'autocorellation va dans le sens de notre intuition précédente : $x_{1,0}$ n'est pas corrélé avec les $x_{1,k}$ pour $k>0$

## Question 2

La classe ImageCompression sert à la fois à résoudre la question 2 et la question 3. 

Nous commençons par créer un objet de la classe ImageCompression que nous appelons "gibbs2".
En paramètre, nous entrons un array numpy de taille 20x20 dont les valeurs vont de 1 à 255 (afin de simuler une image). Cependant, on se restreint à une dimension de 20x20 pour limiter les temps de calculs.



```python
gibbs2 = ImageCompression()
```

La fonction ```gibbs_sampling``` simule la loi des $x_i$, des $\mu_k$ et des $\sigma^2_k$ selon un algorithme de Gibbs Sampling (voir le code pour plus de détails techniques).
On prend comme loi a priori des lois normales pour les $\mu_k$ et des inverse-gamma pour les $\sigma^2_k$ (voir utils.py pour les valeurs exacts des lois à priori)


```python
X, mu, sigma = gibbs2.gibbs_sampling()
```

    100%|██████████| 499/499 [01:16<00:00,  6.56it/s]


Traçons quelques graphes d'auto-corrélation et d'auto-corrélation partielle pour vérifier que nos résultats sont cohérents.


```python
plot_acf(X[2,2,:])
```




    
![png](readme_files/readme_25_0.png)
    




    
![png](readme_files/readme_25_1.png)
    



```python
plot_acf(X[1,2,:])
```




    
![png](readme_files/readme_26_0.png)
    




    
![png](readme_files/readme_26_1.png)
    



```python
plot_acf(X[3,2,:])
```




    
![png](readme_files/readme_27_0.png)
    




    
![png](readme_files/readme_27_1.png)
    


## Question 3

Pour la question 3, on commence par importer nos images.
La méthode convert permet de transformer le chemin d'une image en couleur en l'array numpy associé en noir et blanc. On peut ajouter l'argument "resize" pour resize l'image (par défaut l'image est resize à 128). On resize l'image afin que le temps de calcul soit moins long.


```python
image = Image.open('ensae_image.jpg')
image.show()
```


    
![png](readme_files/readme_30_0.png)
    


Nous allons dans la suite utiliser notre algorithme de compression d'image via la méthode ```compress``` avec differents paramètres pour K et $\beta$

### Première image

#### K=10 et $\beta$=5


```python
compression = ImageCompression(path='ensae_image.jpg', K_potts=10, beta_potts=5)

compressed_meth1 = compression.compress(nb_sample=100)

show(compressed_meth1)
```

    100%|██████████| 99/99 [04:14<00:00,  2.57s/it]



    
![png](readme_files/readme_32_1.png)
    


#### K=20 et $\beta$ = 5


```python
compression = ImageCompression(path='ensae_image.jpg', K_potts=20, beta_potts=5)

compressed_meth1 = compression.compress(nb_sample=100)

show(compressed_meth1)
```

    100%|██████████| 99/99 [07:48<00:00,  4.73s/it]



    
![png](readme_files/readme_34_1.png)
    


#### K=30 et $\beta$ = 5


```python
compression = ImageCompression(path='ensae_image.jpg', K_potts=30, beta_potts=5)

compressed_meth1 = compression.compress(nb_sample=100)

show(compressed_meth1)
```

    100%|██████████| 99/99 [11:45<00:00,  7.12s/it]



    
![png](readme_files/readme_36_1.png)
    


#### K=10 et $\beta$ = 20


```python
compression = ImageCompression(path='ensae_image.jpg', K_potts=10, beta_potts=20)

compressed_meth1 = compression.compress(nb_sample=100)

show(compressed_meth1)
```

    100%|██████████| 99/99 [04:26<00:00,  2.69s/it]



    
![png](readme_files/readme_38_1.png)
    


#### K=10 et $\beta$ = 30


```python
compression = ImageCompression(path='ensae_image.jpg', K_potts=10, beta_potts=30)

compressed_meth1 = compression.compress(nb_sample=100)

show(compressed_meth1)
```

    100%|██████████| 99/99 [04:10<00:00,  2.53s/it]



    
![png](readme_files/readme_40_1.png)
    


### Deuxième image


```python
image = Image.open('paris_image.png')
image.show()
```


    
![png](readme_files/readme_42_0.png)
    


#### K=10 et $\beta$ = 5


```python
compression = ImageCompression(path='paris_image.png', K_potts=10, beta_potts=5)

compressed_meth2 = compression.compress(nb_sample=100)

show(compressed_meth2)
```

    100%|██████████| 99/99 [10:46<00:00,  6.53s/it]



    
![png](readme_files/readme_44_1.png)
    


#### K=20 et $\beta$ = 5


```python
compression = ImageCompression(path='paris_image.png', K_potts=20, beta_potts=5)

compressed_meth2 = compression.compress(nb_sample=100)

show(compressed_meth2)
```

    100%|██████████| 99/99 [22:30<00:00, 13.65s/it]



    
![png](readme_files/readme_46_1.png)
    


#### K=30 et $\beta$ = 5


```python
compression = ImageCompression(path='paris_image.png', K_potts=30, beta_potts=5)

compressed_meth2 = compression.compress(nb_sample=100)

show(compressed_meth2)
```

    100%|██████████| 99/99 [30:38<00:00, 18.57s/it]



    
![png](readme_files/readme_48_1.png)
    


#### K=10 et $\beta$ = 20


```python
compression = ImageCompression(path='paris_image.png', K_potts=10, beta_potts=20)

compressed_meth2 = compression.compress(nb_sample=100)

show(compressed_meth2)
```

    100%|██████████| 99/99 [10:32<00:00,  6.39s/it]



    
![png](readme_files/readme_50_1.png)
    


#### K=10 et $\beta$ = 30


```python
compression = ImageCompression(path='paris_image.png', K_potts=10, beta_potts=30)

compressed_meth2 = compression.compress(nb_sample=100)

show(compressed_meth2)
```

    100%|██████████| 99/99 [10:05<00:00,  6.12s/it]



    
![png](readme_files/readme_52_1.png)
    


Rendez-vous à la soutenance pour les explications plus approfondies ! :)
