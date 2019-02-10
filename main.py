import numpy as np
from math import log
import pandas as pd

from sklearn import linear_model
from AdaptiveHuber import AdaptiveHuber



##########################################################
#
#      Compute table 1 of the paper
#
##########################################################


d = 5
n = 100
beta_star = np.zeros(d)
beta_star[:5] = [5,-10,0,0,3]            


N = 100
algos = [AdaptiveHuber(), linear_model.LinearRegression(), linear_model.Lasso(alpha=1), linear_model.HuberRegressor()]

errors = np.zeros((N, len(algos)))
for k in range(N):
    # Generate data
    X = np.random.multivariate_normal(np.zeros(d), np.identity(d), size = n)
    
    # Generate the tail
    #Eps = np.random.normal(0, 4, size = n) # normal tail
    #Eps = np.random.standard_t(df = 1.5, size = n) # student tail
    Eps = np.random.lognormal(0, 4, size = n) # log-normal tail
    
    Y = np.dot(X, beta_star) + Eps
    
    # Compute L2 errors
    for i, algo in enumerate(algos):
        algo.fit(X, Y)
        errors[k, i] = np.linalg.norm(algo.coef_ - beta_star)
    
errors = pd.DataFrame(errors, columns = ["Adaptive Huber", "OLS", "Lasso", "Huber"])
 
print(errors.describe().loc[["mean", "std"]])


# Comments :
# On a vraiment aps les mêmes résultats que sur le papier même pour les algos de scikit...
# Je vais essayer de tracer les courbes






