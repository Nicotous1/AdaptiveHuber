import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import GridSearchCV
from numba import prange

import warnings


from AdaptiveHuber import AdaptiveHuber, QuantRegScikit, AdaptiveHuberCV
from sklearn.linear_model import LassoCV, Lasso
from sklearn import linear_model

##########################################################
#
#      Compute table 1 of the paper
#
##########################################################

# Comments :
# On a vraiment pas les mêmes résultats que sur le papier même pour les algos de scikit...


def get_errors_for(algos, tails, N, d, n, beta):
    '''
        Generate data from tail and beta and compute error for each pair of algo N times.
        Return a DataFrame of the errors
    '''
    errors = []    
    # Generate features for all algo for this round
    for k in range(N):
        X = np.random.multivariate_normal(np.zeros(d), np.identity(d), size = n)
        
        for tail_name, tail in tails.items():
            print(k, "/", N, " : ", tail_name)
            # Generate data from beta and the tail
            Eps = tail.rvs(n)
            Y = np.dot(X, beta) + Eps
            
            # Run each algo and store the error
            for algo_name, algo in algos.items():
                algo.fit(X, Y)
                error = np.linalg.norm(algo.coef_ - beta)
                errors.append([tail_name, algo_name, error, k])

                
    errors = pd.DataFrame(errors, columns = ["tail", "algo", "l2_error", "round"])
    return errors




n = 500
d = 1000
beta_star = np.zeros(d)
beta_star[:5] = [5,-10,0,0,3]      

adaptiveCV = AdaptiveHuberCV({'c_tau':np.arange(start=0.5,stop=1.5,step=.5), 'c_lamb':np.arange(start=0.005,stop=0.03,step=.005)})
adaptive = AdaptiveHuber(c_lamb=0.005, c_tau=0.5)     
 

N = 100
algos = {"OLS" : linear_model.LinearRegression(),
         "LassoCV" : LassoCV(cv=3),
         "Huber" : linear_model.HuberRegressor(),
         #"MedianReg" : QuantRegScikit(q = 0.5),
         "AdaptiveCV" : adaptiveCV}

tails = {"normal" : stats.norm(loc = 0, scale = 4),
         "student" : stats.t(df = 1.5),
         "lognormal" : stats.lognorm(1, loc = 0, scale = 4)}


errors = get_errors_for(algos, tails, N, d, n, beta_star)

# the table of the paper
table = errors.groupby(["algo", "tail"]).l2_error.describe()[["mean", "std"]]
print(table.round(2))

errors.to_pickle("{}_{}_errors.pickle".format(n, d))

# plot boxplot for algo and tail
errors.boxplot(column = "l2_error", by=["tail", "algo"])



##########################################################
#
#      Phase Transition
#
##########################################################

N = 10
n, d = 500, 1000
beta = np.zeros(d)
beta[:5] = [5,-10,0,0,3]  

# Generate data from beta and the tail
dfs = np.linspace(1.1, 3, 50)

algos = {"OLS" : linear_model.LinearRegression(),
         "Huber" : linear_model.HuberRegressor(),
         "LASSO_CV" : LassoCV(cv=3),
         "Adaptive" : best_adhub}

X = np.random.multivariate_normal(np.zeros(d), np.identity(d), size = n)
errors = []
for df in dfs:
    print(df, "...")
    for k in prange(N):
        Eps = np.random.standard_t(df, size = n)
        Y = np.dot(X, beta) + Eps
        for algo_name, algo in algos.items():
            algo.fit(X, Y)
            if algo_name!="Adaptive":
                error = np.linalg.norm(algo.coef_ - beta)
            else:
                error = np.linalg.norm(algo.best_estimator_.coef_ - beta)
            errors.append([df, error, k, algo_name])
    
errors = pd.DataFrame(errors, columns = ["df", "l2_error", "round", "algo"])
errors["delta"] = errors["df"] -1 -0.05
errors["log_error"] = -np.log(errors.l2_error)


errors.loc[errors.tail == "normal"].groupby(["delta", "algo"]).log_error.mean().unstack().plot()



##########################################################
#
#      Accuracy of the selection
#
##########################################################




    
    

































































