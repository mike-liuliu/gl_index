 

def got_X_divide_from_label(X, label):
    X_divide = []
    for jj, ii in enumerate(list(set(label))):
        assert jj == ii, "jj == ii"
        ppp = X[label == ii]
        X_divide.append(ppp)
    return X_divide


def normalize_label(label):
    kk = set(label)
    dic = {}
    for i, uu in enumerate(kk):
        dic[uu] = i
    new_label = []
    for tt in label:
        new_label.append(dic[tt])
    return np.array(new_label)  
 

def not_good_label(label):
    kk = set(label)
    if (min(kk) != 0) or (max(kk) != len(kk) - 1):
        return True
    return False
 
    
 
def cal_one_data_index_list(X, index_func, smaller_better, data_n, multiple_label):
    one_data_index_list = []

    for k,label in enumerate(multiple_label):
        # print(k)
        label = np.array(label)
        if smaller_better:
            index_value = index_func(X,label)
        else:
            index_value = -index_func(X,label)

        one_data_index_list.append(index_value)   
    return one_data_index_list
    
def plot_first_n_label_by_index(one_data_index_list, multiple_label, X, true_label_position,smaller_better, succeeded_only):
    # global kk
    true_label = multiple_label[true_label_position]

    ii = np.argmin(one_data_index_list)
    this_label = np.array(multiple_label[ii])
    AR_best = adjusted_rand_score(this_label,true_label)            
    AR_best = np.round(AR_best, 5)
    
    if succeeded_only and AR_best < 0.95:
        return AR_best
    
    fig = plt.figure(figsize=(14.5, 2.6), constrained_layout=True)
    spec = fig.add_gridspec(1, 5,hspace=0.1)
    
    a,b = X.shape

    for tt, ii in enumerate(np.argsort(one_data_index_list)[:5]):
        this_label = np.array(multiple_label[ii])
        adjusted_rand_gra = adjusted_rand_score(this_label,true_label)            
        adjusted_rand_gra = np.round(adjusted_rand_gra, 3)
 
        scorr2 = np.sort(one_data_index_list)[tt]
        scorr2 = np.round(scorr2, 5)
        if not smaller_better:
            scorr2 = - scorr2
        if b == 2:
            ax10 = fig.add_subplot(spec[0, tt])
        if b == 3:
            ax10 = fig.add_subplot(spec[0, tt], projection='3d')              
        
        K = len(set(this_label))


        plot_2D_or_3D_data_axes(X,  this_label ,ax10)
        ax10.set_title(f"K={K}, S={scorr2}, AR={adjusted_rand_gra}") 
#     plt.savefig(f'./da/img/{kk}.png')
    plt.show()
    
    return AR_best
    
def plot_2D_or_3D_data_axes(data, label,plt_k):
    a,b = data.shape
    markers = ["." , "+", "s" , "x", "v" , "1" , "p", "P", "*", "o" , "d"]
    if b == 2:
        if label is not None:
            X_divide = got_X_divide_from_label(data, label)
            for tt in range(len(X_divide)):
                plt_k.scatter(X_divide[tt][:,0],X_divide[tt][:,1], marker=markers[tt%len(markers)])
        else:
            plt_k.scatter(data[:, 0], data[:, 1])
    elif b == 3:
        if label is not None:
            X_divide = got_X_divide_from_label(data, label)
            for tt in range(len(X_divide)):
                plt_k.scatter(X_divide[tt][:,0],X_divide[tt][:,1], X_divide[tt][:,2], marker=markers[tt%len(markers)])                        
        else:
            plt_k.scatter(data[:, 0], data[:, 1], data[:, 2])
    else:
        raise ValueError("Not 2D or 3D!")

def index_plot_first_n_label_one_data(index_func, smaller_better, data_id, succeeded_only):
    global multiple_label_145, test_data_145, true_label_position_145
    multiple_label = multiple_label_145[data_id]
    X = test_data_145[data_id]    
    true_label_position = true_label_position_145[data_id]

    one_data_index_list = cal_one_data_index_list(X, index_func, smaller_better, data_id, multiple_label)
    
    AR_best = plot_first_n_label_by_index(one_data_index_list, multiple_label, X, true_label_position,smaller_better, succeeded_only)
    return AR_best 

    
    
    
    
    
    
    
    
    