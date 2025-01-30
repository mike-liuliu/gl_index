import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics.cluster import adjusted_rand_score
 
import os
from ctypes import cdll, c_char_p
import numpy.ctypeslib as ctl
from sklearn.metrics import pairwise_distances
import ctypes
 
from sklearn import metrics
from cdbw import CDbw
import kmedoids
 


def plot_2D_or_3D_data_empty_circles_weak_strong_multi_one_scom(data, labels,centers_idx, strong_ambi_p_idx, weak_ambi_p_idx):
# weak_ambi_p has label num_clusters + 1, strong ambi_p has label num_clusters.
    a,b = data.shape
    if centers_idx is not None:
        centers_idx = centers_idx.copy()
        centers_idx = [ii[0] for ii in centers_idx]
        
    ambi_p_idx = strong_ambi_p_idx + weak_ambi_p_idx
    
    non_ambi_p_idx = [ii for ii in range(a) if ii not in ambi_p_idx]
    
    new_data = data[non_ambi_p_idx]
    new_labels = labels[non_ambi_p_idx]
    
    
        
    markers = ["." , "+", "s" , "x", "v" , "1" , "p", "P", "*", "o" , "d"]
    if b == 2:
        if labels is not None:
            X_divide = got_X_divide_from_labels(new_data, new_labels)
            for tt in range(len(X_divide)):
                plt.scatter(X_divide[tt][:,0],X_divide[tt][:,1], marker=markers[tt%len(markers)])
            if strong_ambi_p_idx:
                plt.scatter(data[strong_ambi_p_idx][:,0],data[strong_ambi_p_idx][:,1], facecolors='k', color = 'k', s=15)
            if weak_ambi_p_idx:
                plt.scatter(data[weak_ambi_p_idx][:,0],data[weak_ambi_p_idx][:,1], facecolors='none', color = 'k', s=15)
            plt.scatter(data[centers_idx][:,0], data[centers_idx][:,1], c ="r",  marker= "*")
            plt.show()                        
 
    elif b == 3:
        fig = plt.figure()
        adata = fig.add_subplot(111, projection='3d')
        if labels is not None:
            X_divide = got_X_divide_from_labels(new_data, new_labels)
            for tt in range(len(X_divide)):
                adata.scatter(X_divide[tt][:,0],X_divide[tt][:,1], X_divide[tt][:,2],marker=markers[tt%len(markers)]) 
            if strong_ambi_p_idx:
                adata.scatter(data[strong_ambi_p_idx][:,0],data[strong_ambi_p_idx][:,1], data[strong_ambi_p_idx][:,2], facecolor='k', edgecolor='k', s=15)
            if weak_ambi_p_idx:
                adata.scatter(data[weak_ambi_p_idx][:,0],data[weak_ambi_p_idx][:,1], data[weak_ambi_p_idx][:,2], facecolor=(0,0,0,0), edgecolor='k', s=15)
            adata.scatter(data[centers_idx][:,0], data[centers_idx][:,1],data[centers_idx][:,2], c ="r",  marker= "*")             
            plt.show()

def got_X_divide_from_labels(X, labels):
    X_divide = []
    for jj, ii in enumerate(list(set(labels))):
        assert jj == ii, "jj == ii"
        ppp = X[labels == ii]
        X_divide.append(ppp)
    return X_divide

def test_mmj_kmeans_multi_one_scom(data_id, datasets, datasets_true_K, attempts = 20):
    X = datasets[data_id] 
    num_clusters = datasets_true_K[data_id]

    # mmj_matrix = pickle.load( open( f"./mmj_distance_matrix_precomputed/mmj_r_data_{data_id}.p", "rb" ) ) 

    # Calculate MMJ distance matrix with Algorithm 4, implementation and testing of Algorithm 4 see another paper:
    # http://www.arxiv.org/abs/2407.07058
    mmj_matrix =  cal_mmj_matrix_by_algo_4_Calculation_and_Copy(X) 

    label, centers_idx, loss, strong_ambi_p_idx, weak_ambi_p_idx = KMeans_several_times_ambi_points_multi_one_scom(X, num_clusters, attempts, mmj_matrix)
 
 
    if strong_ambi_p_idx or weak_ambi_p_idx:
        plot_2D_or_3D_data_empty_circles_weak_strong_multi_one_scom(X, label,centers_idx, strong_ambi_p_idx, weak_ambi_p_idx)        
    else:
        plot_2D_or_3D_data(X, label, centers_idx = centers_idx, plot_center = 1)
            
            
    # return len(X), len(strong_ambi_p_idx) + len(weak_ambi_p_idx)

def plot_2D_or_3D_data(data, labels, * ,plot_center = False, centers_idx = None):
    # pdb.set_trace()
    
    if centers_idx is not None:
        if type(centers_idx[0]) is not int:
            centers_idx = centers_idx.copy()
            centers_idx = [ii[0] for ii in centers_idx]
    a,b = data.shape
    markers = ["." , "+", "s" , "x", "v" , "1" , "p", "P", "*", "o" , "d"]
    if b == 2:
        if labels is not None:
            X_divide = got_X_divide_from_labels(data, labels)
            for tt in range(len(X_divide)):
                plt.scatter(X_divide[tt][:,0],X_divide[tt][:,1], marker=markers[tt%len(markers)])
                       
            if plot_center:
                plt.scatter(data[centers_idx][:,0], data[centers_idx][:,1], c ="r",  marker= "*")
            plt.show()
        else:
            plt.scatter(data[:, 0], data[:, 1])
            plt.show()
    elif b == 3:
        fig = plt.figure()
        adata = fig.add_subplot(111, projection='3d')
        if labels is not None:
            X_divide = got_X_divide_from_labels(data, labels)
            for tt in range(len(X_divide)):
                adata.scatter(X_divide[tt][:,0],X_divide[tt][:,1], X_divide[tt][:,2],marker=markers[tt%len(markers)])
            if plot_center:
                adata.scatter(data[centers_idx][:,0], data[centers_idx][:,1], data[centers_idx][:,2], c ="r",   marker= "*") 
            plt.show()
        else:
            adata.scatter(data[:, 0], data[:, 1], data[:, 2])
            plt.show()
    else:
        print("Not 2D or 3D!")
        # pass




def cal_one_data_index_list(X, index_func, smaller_better, data_n, multiple_labels):
    one_data_index_list = []

    for k,labels in enumerate(multiple_labels):
        # print(k)
        labels = np.array(labels)
        if smaller_better:
            index_value = index_func(X,labels)
        else:
            index_value = -index_func(X,labels)

        one_data_index_list.append(index_value)   
    return one_data_index_list

def cal_centroid_X( X_id, dis_matrix): 
    
    distance_matrix_square = dis_matrix**2
  
    square_dis_list = [sum(distance_matrix_square[pp,X_id]) for pp in X_id]    
    center_idx = np.argmin(square_dis_list)

    return center_idx

def cal_centroid_id( X, labels, mmj_matrix): 
#     import pdb;pdb.set_trace()
    n_labels = len(set(labels))
    
    X_id = np.array(range(len(X)))
 
    distance_matrix_square = mmj_matrix**2
    center_idx = []

    for kkk in range(n_labels):
 
        clu_index = X_id[labels == kkk]
        square_dis_list = [sum(distance_matrix_square[pp,clu_index]) for pp in clu_index]
 
        mmm = clu_index[np.argmin(square_dis_list)]         
        center_idx.append(mmm)
        
    return center_idx

def plot_first_n_label_by_index(one_data_index_list, multiple_labels, X, true_label_position,smaller_better, succeeded_only):
    # global kk
    true_labels = multiple_labels[true_label_position]

    ii = np.argmin(one_data_index_list)
    this_labels = np.array(multiple_labels[ii])
    AR_best = adjusted_rand_score(this_labels,true_labels)            
    AR_best = np.round(AR_best, 5)
    
    if succeeded_only and AR_best < 0.95:
        return AR_best
    
    fig = plt.figure(figsize=(14.5, 2.6), constrained_layout=True)
    spec = fig.add_gridspec(1, 5,hspace=0.1)
    
    a,b = X.shape

    for tt, ii in enumerate(np.argsort(one_data_index_list)[:5]):
        this_labels = np.array(multiple_labels[ii])
        adjusted_rand_gra = adjusted_rand_score(this_labels,true_labels)            
        adjusted_rand_gra = np.round(adjusted_rand_gra, 3)
 
        scorr2 = np.sort(one_data_index_list)[tt]
        scorr2 = np.round(scorr2, 5)
        if not smaller_better:
            scorr2 = - scorr2
        if b == 2:
            ax10 = fig.add_subplot(spec[0, tt])
        if b == 3:
            ax10 = fig.add_subplot(spec[0, tt], projection='3d')              
        
        K = len(set(this_labels))


        plot_2D_or_3D_data_axes(X,  this_labels ,ax10)
        ax10.set_title(f"K={K}, S={scorr2}, AR={adjusted_rand_gra}") 
#     plt.savefig(f'./da/img/{kk}.png')
    plt.show()
    
    return AR_best
    
def plot_2D_or_3D_data_axes(data, labels,plt_k):
    a,b = data.shape
    markers = ["." , "+", "s" , "x", "v" , "1" , "p", "P", "*", "o" , "d"]
    if b == 2:
        if labels is not None:
            X_divide = got_X_divide_from_labels(data, labels)
            for tt in range(len(X_divide)):
                plt_k.scatter(X_divide[tt][:,0],X_divide[tt][:,1], marker=markers[tt%len(markers)])
        else:
            plt_k.scatter(data[:, 0], data[:, 1])
    elif b == 3:
        if labels is not None:
            X_divide = got_X_divide_from_labels(data, labels)
            for tt in range(len(X_divide)):
                plt_k.scatter(X_divide[tt][:,0],X_divide[tt][:,1], X_divide[tt][:,2], marker=markers[tt%len(markers)])                        
        else:
            plt_k.scatter(data[:, 0], data[:, 1], data[:, 2])
    else:
        raise ValueError("Not 2D or 3D!")

def index_plot_first_n_label_one_data(index_func, smaller_better, data_id, succeeded_only):
    global multiple_label_145, test_data_145, true_label_position_145
    multiple_labels = multiple_label_145[data_id]
    X = test_data_145[data_id]    
    true_label_position = true_label_position_145[data_id]

    one_data_index_list = cal_one_data_index_list(X, index_func, smaller_better, data_id, multiple_labels)
    
    AR_best = plot_first_n_label_by_index(one_data_index_list, multiple_labels, X, true_label_position,smaller_better, succeeded_only)
    return AR_best 

def mmj_calinski_harabasz_score(X, labels, ignor_less_than_n = 1):
    global dis_matrix
 
    n_samples = len(X)
    n_labels = len(set(labels))
    
    X_id = np.array(range(len(X)))
    
 
    extra_disp, intra_disp = 0.0, 0.0
    mean = cal_centroid_X( X_id, dis_matrix)
    
    centroids = cal_centroid_id( X, labels, dis_matrix)
    
    for k in range(n_labels):
        cluster_k = X_id[labels == k]

        #if a cluster contains less than ignor_less_than_n points, we just return the worst  calinski harabasz index score.
        if len(cluster_k) < ignor_less_than_n:
            return -np.inf

        mean_k = centroids[k]
        extra_disp += len(cluster_k) * (dis_matrix[mean,mean_k] ** 2)
#         import pdb;pdb.set_trace()
        intra_disp += np.sum(dis_matrix[cluster_k,mean_k] ** 2)

    return (
        1.0
        if intra_disp == 0.0
        else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
    )

def Silhouette_coefficient(X, label):
    return metrics.silhouette_score(X, label)

def calinski_harabasz_index(X, label):
    return metrics.calinski_harabasz_score(X, label)

def davies_bouldin_index(X, label):
    return metrics.davies_bouldin_score(X, label)

def cal_CDbw(X, label):
    return CDbw(X, label, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False)
 
def mmj_Silhouette_coefficient(X, label, use_scikit = True):
    global dis_matrix
    if use_scikit:
        return metrics.silhouette_score(dis_matrix, label, metric='precomputed')

def mmj_davies_bouldin_score(X, labels):
    global dis_matrix
 
    n_samples = len(X)
    n_labels = len(set(labels))
    
    X_id = np.array(range(len(X)))
  
    intra_dists = np.zeros(n_labels)
    centroids = cal_centroid_id( X, labels, dis_matrix)
    
    for k in range(n_labels):
        cluster_k = X_id[labels == k]
 
        centroid = centroids[k]
    
        temp_dis = pairwise_dist_from_id(cluster_k, [centroid], dis_matrix)
 
        intra_dists[k] = np.average(temp_dis)

    centroid_distances = pairwise_dist_from_id(centroids, centroids, dis_matrix)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0
#     import pdb;pdb.set_trace()
    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def pairwise_dist_from_id(X_id, centroid_id, mmj_matrix): 
#     import pdb;pdb.set_trace()
    m, n = len(X_id), len(centroid_id)
    dists = np.zeros((m,n))        
    for i in range(m):
        for j in range(n):
            p = X_id[i]
            q = centroid_id[j]
            dists[i,j] = mmj_matrix[p,q]

    return dists

