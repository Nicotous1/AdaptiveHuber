import numpy as np
from math import log
from sklearn.base import BaseEstimator, RegressorMixin





##########################################################
#
#      Utilities for Huber method
#
##########################################################



def S(X, lamb):
    '''
        Soft-thresholding function
    '''
    return np.sign(X)*np.maximum(np.abs(X) - lamb, np.zeros(X.shape))



def g(X, Y, beta_next, beta, tau, phi):
    '''
        g function of the paper (formula of part 5)
    '''
    return   huber_loss(X, Y, beta, tau)\
           + np.sum(huber_loss_grad(X, Y, beta, tau) * (beta_next - beta))\
           + phi/2*np.linalg.norm(beta_next - beta)**2



def huber_loss(X, Y, beta, tau):
    '''
        Huber loss for the sample (X,Y)
        Compute the linear and the square loss and then do the sum
    '''
    n = X.shape[0]
    eps_abs = np.abs(Y - np.dot(X, beta))
    inf_selector = eps_abs < tau
    square_loss = np.sum((eps_abs[inf_selector]**2)/2)
    linear_loss = np.sum(tau * eps_abs[~inf_selector] - (tau**2)/2)
    return (square_loss + linear_loss)/n



def huber_loss_grad(X, Y, beta, tau):
    '''
        Compute the huber loss gradient for a sample (X, Y) with the matrix formula
    '''
    n, d = X.shape
    inf_selector = np.abs(Y - np.dot(X, beta)) < tau
    
    grad_power = np.dot(X[inf_selector].T, (np.dot(X[inf_selector], beta) - Y[inf_selector]))
    
    grad_linear = - tau * np.dot(X[~inf_selector].T, np.sign(Y[~inf_selector] - np.dot(X[~inf_selector], beta)))
    
    grad = (grad_power + grad_linear)/n
    return grad



from scipy import optimize
def huber_loss_grad_approx(X, Y, beta, tau):
    '''
        Compute an approximate gradient of the huber loss
        To check that it gives the same thing than the matrix formula
    '''
    n, d = X.shape
    def f(x):
        return huber_loss(X, Y,  x, tau)

    return optimize.approx_fprime(beta, f, [0.000001]*d)



def huber_loss_grad_slow(X, Y, beta, tau):
    '''
        Compute the gradient coordinates by coordinates
        A lot slower but easier to check
    '''
    n, d = X.shape
    grad = np.zeros(d)
    for i in range(n):
        x = Y[i] - np.sum(X[i]*beta)
        if abs(x) <= tau:
            for j in range(d):
                grad[j] += X[i,j]*(-x)
        else:
            for j in range(d):
                grad[j] += -np.sign(x)*tau*X[i,j]
    return grad/n

    





##########################################################
#
#      Estimator class
#
##########################################################



class AdaptiveHuber(BaseEstimator, RegressorMixin):
    '''
        Adaptive Huber from the paper of Sun & al
    '''
    
    def __init__(self, c_tau = 0.5, c_lamb = 0.005, gamma = 2, phi = 1, epsilon = 1E-6):
        '''
            c_tau and c_lamb are parameters to be set via cross validation
        '''
        self.c_tau = c_tau
        self.c_lamb = c_lamb
        self.gamma = gamma
        self.phi = phi
        self.epsilon = epsilon
    
    
    
    def fit(self, X, y=None):
        '''
         Scikit-learn function to fit but it just use the _LAMM function below
        '''
        self.coef_ = self._LAMM(X, y)
        return self
    
    
    
    def predict(self, X):
        '''
            return Y for the sample (X)
        '''
        return np.dot(X, self.coef_)
        
        
        
    def _LAMM(self, X, Y):
        '''
            LAAM algorithm for regularized adaptive Huber regression
        '''
        # Shortcut parameters
        n, d = X.shape
        phi, gamma, epsilon = self.phi, self.gamma, self.epsilon
        
        # Adaptive parameters
        conf_level = np.std(Y)*(n/log(n))**0.5
        tau = self.c_tau * conf_level
        lamb = self.c_lamb * conf_level
        
        # Iterate beta until convergence according to epsilon
        beta = np.zeros(d)
        for _ in range(50):
            # Compute the next beta
            grad = huber_loss_grad(X, Y, beta, tau)
            beta_next = S(beta - grad/phi, lamb/phi)
        
            # Check if phi should be increase
            g_k, L_k = g(X, Y, beta_next, beta, tau, phi), huber_loss(X, Y, beta_next, tau)
            if g_k < L_k:
                phi *= gamma
                
            # Check if convergence happened
            delta = np.linalg.norm(beta_next - beta)
            if delta < epsilon:
                return beta_next
            beta = beta_next
                
        return beta        
        
    