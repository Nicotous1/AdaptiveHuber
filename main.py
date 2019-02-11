import numpy as np
from math import log
import pandas as pd

from sklearn import linear_model
from AdaptiveHuber import AdaptiveHuber
from scipy import stats



##########################################################
#
#      Compute table 1 of the paper
#
##########################################################

# Comments :
# On a vraiment aps les mêmes résultats que sur le papier même pour les algos de scikit...


def get_errors_for(algos, tails, N, d, n, beta):
    '''
        Generate data from tail and beta and compute error for each pair of algo N times.
        Return a DataFrame of the errors
    '''
    errors = []
    # Generate features for all algo for this round
    X = np.random.multivariate_normal(np.zeros(d), np.identity(d), size = n)
    for k in range(N):
        
        for tail_name, tail in tails.items():
            # Generate data from beta and the tail
            Eps = tail.rvs(n)
            Y = np.dot(X, beta_star) + Eps
            
            # Run each algo and store the error
            for algo_name, algo in algos.items():
                algo.fit(X, Y)
                error = np.linalg.norm(algo.coef_ - beta_star)
                errors.append([tail_name, algo_name, error, k])
                
    errors = pd.DataFrame(errors, columns = ["tail", "algo", "l2_error", "round"])
    return errors


d = 5
n = 100
beta_star = np.zeros(d)
beta_star[:5] = [5,-10,0,0,3]            


N = 300
algos = {"OLS" : linear_model.LinearRegression(),
         "LASSO" : linear_model.Lasso(alpha=1),
         "Huber" : linear_model.HuberRegressor(),
         "Adaptive" : AdaptiveHuber()}

tails = {"normal" : stats.norm(loc = 0, scale = 4),
         "student" : stats.t(df = 1.5),
         "lognormal" : stats.lognorm(1, loc = 0, scale = 4)}


errors = get_errors_for(algos, tails, N, d, n, beta_star)

# the table of the paper
table = errors.groupby(["algo", "tail"]).l2_error.describe()[["mean", "std"]]
print(table)

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
         "Adaptive" : AdaptiveHuber()}

X = np.random.multivariate_normal(np.zeros(d), np.identity(d), size = n)
errors = []
for df in dfs:
    print(df, "...")
    for k in range(N):
        Eps = np.random.standard_t(df, size = n)
        Y = np.dot(X, beta) + Eps
        for algo_name, algo in algos.items():
            algo.fit(X, Y)
            error = np.linalg.norm(algo.coef_ - beta)
            errors.append([df, error, k, algo_name])
    
errors = pd.DataFrame(errors, columns = ["df", "l2_error", "round", "algo"])
errors["delta"] = errors["df"] -1 -0.05
errors["log_error"] = -np.log(errors.l2_error)


errors.groupby(["delta", "algo"]).log_error.mean().unstack().plot()
















import matplotlib.pyplot as plt

plt.figure()
plt.plot(dfs - 1 - 0.05, -np.log(errors))
    
    
    

































































