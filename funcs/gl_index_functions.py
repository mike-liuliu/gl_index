from sklearn import mixture 
from multiprocessing import Value 
from sklearn.metrics.cluster import rand_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics.cluster import adjusted_rand_score

 


# We need to fit a best bandwidth for data X, using kernel density estimation, the fitting only need to do once. 
# There is no need to fit the X_best_bandwidth multiple times, so we leave it to the outside of the index function.
def gl_index(X,label):
    global X_best_bandwidth
    alpha = 4.0
    delta = [0.75, 0.25]
    delta_e = 1000.0
    
    N = X.shape[0]
 
    if not_good_label(label):
        label = normalize_label(label)
        
    label = np.array(label)
 
    X_divide = got_X_divide_from_label(X, label) 
    
    K = len(X_divide)
    
    loglike_X_matrix_K_by_N = []
    min_max_list = []
    total_sum = 0
  
    
    for each_divide in X_divide:
        density_estima = KernelDensity(**X_best_bandwidth)
        density_estima.fit(each_divide)
        loglike_X = density_estima.score_samples(X)  
        loglike_X_matrix_K_by_N.append(loglike_X)
        
 
        log_likely = density_estima.score_samples(each_divide)
        min_kde = np.min(log_likely)
        max_kde = np.max(log_likely)
        stddd = np.std(log_likely)
        delta_q = alpha * stddd
        if delta_q == 0:
            delta_q = delta_e
        min_max_list.append([min_kde - delta_q, max_kde + delta_q])
        
        likely = np.exp(log_likely)
        ratios = likely/np.max(likely)
        total_sum += np.sum(ratios)
 
        
    loglike_X_matrix_K_by_N = np.array(loglike_X_matrix_K_by_N)
    min_max_list = np.array(min_max_list)
    
    I_a = cal_I_a(N, K, loglike_X_matrix_K_by_N, min_max_list)
    I_s = 1 - total_sum/N
 
    index = delta[0]*I_a + delta[1]*I_s
    return index

def cal_I_a(N, K, loglike_X_matrix_K_by_N, min_max_list): 
    counnn = np.zeros(N)
    ambi_index = []

    for i in range(N):
        for j in range(K):          
            if loglike_X_matrix_K_by_N[j, i] >= min_max_list[j,0] and loglike_X_matrix_K_by_N[j,i] <= min_max_list[j,1]: 
                counnn[i] += 1 
                if counnn[i] > 1:
                    ambi_index.append(i)
                    break
 
    I_a = len(ambi_index)/N
    return I_a
 

def kde_select_best_bandwidth_sliding_window(X):

    if len(X) == 1:
        best_bandwidth = {"bandwidth":  1.0}
        return best_bandwidth
     
    last_best_bandwidth = -1      
    n = 0   
    num_of_bandwidths = 10
    stepsize = 0.1
 
    while(True):
            logspace = np.logspace(n-stepsize, n, num_of_bandwidths)
            logspace = np.round(logspace, 10)            
            global_best_bandwidth = kde_select_best_bandwidth(X,logspace)
            
            if np.abs(global_best_bandwidth["bandwidth"] - last_best_bandwidth) <= 1e-7:            
                return global_best_bandwidth 
            
            last_best_bandwidth = global_best_bandwidth["bandwidth"]
  
            if global_best_bandwidth["bandwidth"] == np.min(logspace):
                n -= stepsize
                n = np.round(n, 2)
                assert n > -10, "n > -10"                   
            elif global_best_bandwidth["bandwidth"] == np.max(logspace):
                n += stepsize
                n = np.round(n, 2)
                assert n < 10, "n < 10"
            else:
                return global_best_bandwidth
            
def kde_select_best_bandwidth(X,logspace):
    if len(X) < 5:
        cv = len(X)
    else:
        cv = 5
    cv_KFold = KFold(n_splits = cv, shuffle=True)  
    params = {"bandwidth":  logspace}
    grid = GridSearchCV(KernelDensity(), params, cv=cv_KFold, refit=False)
    grid.fit(X)
    best_bandwidth = grid.best_params_
    return best_bandwidth  

# In the "gl_index_Bayesian_GMM" function, we use Bayesian Gaussian Mixture to do the density estimation. Readers can test other density estimation methods.
def gl_index_Bayesian_GMM(X,label):
    alpha = 1e-06
    delta = [0.62, 0.38]
    delta_e = 1000.0
    
    N = X.shape[0]
 
    if not_good_label(label):
        label = normalize_label(label)
        
    label = np.array(label)
 
    X_divide = got_X_divide_from_label(X, label) 
    
    K = len(X_divide)
    
    loglike_X_matrix_K_by_N = []
    min_max_list = []
    total_sum = 0
  
    
    for each_divide in X_divide:
        density_estima = density_estima_BayesianGaussianMixture(each_divide)

        loglike_X = density_estima.score_samples(X)  
        loglike_X_matrix_K_by_N.append(loglike_X)
 
        log_likely = density_estima.score_samples(each_divide)
        min_kde = np.min(log_likely)
        max_kde = np.max(log_likely)
        stddd = np.std(log_likely)
        delta_q = alpha * stddd
        if delta_q == 0:
            delta_q = delta_e
        min_max_list.append([min_kde - delta_q, max_kde + delta_q])
        
        likely = np.exp(log_likely)
        ratios = likely/np.max(likely)
        total_sum += np.sum(ratios)
 
        
    loglike_X_matrix_K_by_N = np.array(loglike_X_matrix_K_by_N)
    min_max_list = np.array(min_max_list)
    
    I_a = cal_I_a(N, K, loglike_X_matrix_K_by_N, min_max_list)
    I_s = 1 - total_sum/N
 
    index = delta[0]*I_a + delta[1]*I_s
    return index

def density_estima_BayesianGaussianMixture(each_divide):
 
    random.seed(242)
    np.random.seed(242)

    if len(each_divide) <= 2:
        params = {"bandwidth":  1.0}
        density_estima = KernelDensity(**params)
        density_estima.fit(each_divide)        
    
    elif len(each_divide) <= 10:
        density_estima = mixture.BayesianGaussianMixture(n_components=2, covariance_type="full")
        density_estima.fit(each_divide)

    elif len(each_divide) <= 20:
        density_estima = mixture.BayesianGaussianMixture(n_components=5, covariance_type="full")
        density_estima.fit(each_divide)                
    elif len(each_divide) <= 50:
        density_estima = mixture.BayesianGaussianMixture(n_components=8, covariance_type="full")
        density_estima.fit(each_divide)  
    elif len(each_divide) <= 100:
        density_estima = mixture.BayesianGaussianMixture(n_components=10, covariance_type="full")
        density_estima.fit(each_divide) 
    elif len(each_divide) <= 200:
        density_estima = mixture.BayesianGaussianMixture(n_components=15, covariance_type="full")
        density_estima.fit(each_divide)       
    else:
        density_estima = mixture.BayesianGaussianMixture(n_components=30, covariance_type="full")
        density_estima.fit(each_divide)                 
 
    return density_estima




    
    
    
    
    
    
    
    
    