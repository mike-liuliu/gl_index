# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:32:29 2022

@author: Ali Şenol

if you use the code, please cite the article given below:
    
    Ali Şenol, "VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped 
    Clusters Based on the Kernel Density Estimation", Computational Intelligence and Neuroscience, 
    vol. 2022, Article ID 4059302, 20 pages, 2022. https://doi.org/10.1155/2022/4059302
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import random
from scipy.spatial import KDTree
import scipy.io
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from IPython import get_ipython
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
# get_ipython().magic('reset -sf')
# get_ipython().magic('clear all -sf')

def closest_node(n, v):
    kdtree = KDTree(v)
    d, i = kdtree.query(n)
    return d

def VIASCKDE(X, labels,krnl='gaussian', b_width=0.05):  
    CoSeD=np.array([],[])
    num_k = np.unique(labels)
    kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    ASC=np.array([])
    numC=np.array([])
    CoSeD=np.array([])
    viasc=0
    if len(num_k)>1:      
        for i in num_k:
            data_of_cluster=X[labels==i]    
            data_of_not_its=X[labels!=i]
            isos=iso[labels==i]
            isos = (isos-min(isos))/(max(isos)-min(isos))
            for j in range(len(data_of_cluster)): # for each data of cluster j
                row=np.delete(data_of_cluster,j,0) # exclude the data j 
                XX=data_of_cluster[j]
                a=closest_node(XX, row)
                b=closest_node(XX, data_of_not_its)
                ASC=np.hstack((ASC,((b-a)/max(a,b)) * isos[j]))
            numC=np.hstack((numC,ASC.size))
            CoSeD=np.hstack((CoSeD,ASC.mean())) 
        for k in range(len(numC)):
            viasc+=numC[k]*CoSeD[k];
        viasc=viasc/sum(numC)
        # print("viasc=%0.4f"%viasc)
    else:
        viasc=float("nan")
    return viasc
def plotGraph(data, labels):
    plt.rcParams['figure.dpi'] = 100
    plt.subplots_adjust(left=.2, right=.98, bottom=.001, top=.96, wspace=.05,
                hspace=.01)
    plt.rcParams["figure.figsize"] = (4.2,4)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors,edgecolors='k')

