import numpy as np
from scipy.stats import zscore
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import scipy.io
import scipy.linalg
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


#n = int(sys.argv[1])
#print(n)

user_list = [100,101,102,103,104,105,106,107,108,109,10,110,111,112,113,114,115,116,11,12,13,14,15,16,17,18,19,1,20,21,22,23,24,25,26,27,28,29,2,30,
             31,32,33,34,35,36,37,38,39,3,40,41,42,43,44,45,46,47,48,49,4,50,51,52,53,54,55,56,57,58,59,5,60,61,62,63,64,65,66,67,68,69,70,71,72,73,
             74,75,76,77,78,79,7,80,81,82,83,84,85,86,87,88,89,8,90,91,92,93,94,95,96,97,98,99,9]

common_features =  ['mean_F1', 'mean_F2', 'mean_F3', 'mean_F4', 'Tri_graph', 'Error_rate_%','neg_UD_%', 'mean_hold_time',
                    'mean_F1_dis_1_LL','mean_F1_dis_2_LL', 'mean_F1_dis_2_RR', 'mean_F1_dis_3_LL','mean_F1_dis_3_RR', 
                    'mean_F2_dis_1_LL','mean_F2_dis_2_LL', 'mean_F2_dis_2_RR', 'mean_F2_dis_3_LL','mean_F2_dis_3_RR', 
                    'mean_F3_dis_2_RR', 'mean_F3_dis_3_LL','mean_F4_dis_1_LL','mean_F4_dis_2_LL', 'mean_F4_dis_2_RR',
                    'mean_F4_dis_3_LL','mean_F1_se', 'mean_F2_se', 'mean_F3_se','mean_F4_se', 'mean_F1_th', 'mean_F2_th', 
                    'mean_F3_th', 'mean_F4_th']

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K

def MyTCA(Xs, Xt, kernel_type, gamma, lamb, dim):
        '''
        $--------  Need to Modify the description ---------$
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(kernel_type, X, None, gamma=gamma)
        n_eye = m if kernel_type == 'primal' else n
        a, b = K @ M @ K.T + lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        
        A = V[:, ind[:dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        
        return Xs_new, Xt_new

def jdrf(x_train, y_train, n_trees=30):
    param_grid = {'bootstrap': [False, True],
              'max_depth': [30,40],
              'min_samples_leaf': [2,4],
              'min_samples_split': [5,6],
              'n_estimators': [500, 1000],
              'criterion':['gini']
             }

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, refit = True,
                           cv = cv_inner, n_jobs = 1, scoring = 'roc_auc')
    
    tuning = grid_rf.fit(x_train, y_train)

    best_hyperparams = tuning.best_params_

    best_model = RandomForestClassifier(random_state=42, **best_hyperparams)

    trained_model = best_model.fit(x_train, y_train)
    
    return trained_model


source_data = pd.read_csv('C:/Users/s3929438/all_features_mobile_100_all_final_latest.csv')
#target_data = pd.read_csv('C:/Users/s3929438/all_features_mobile_100_all_final_latest.csv')
target_data = pd.read_csv('C:/Users/s3929438/all_features_tablet_100_all_latest.csv')

user_id_list = []
acc_list = []

for index, i in enumerate (user_list):
    random.seed(42)
    np.random.seed(42)

    dim=len(common_features)
    user_id = i

    src_user = source_data[source_data['User']==user_id]
    src_features = src_user[common_features]
    src_features.insert(loc = len(src_features.columns),column = 'User_type',value = 1)

    src_other_users = source_data[source_data['User']!=user_id]
    src_sample_others = src_other_users.loc[np.random.RandomState(seed=42).choice(src_other_users.index, size=len(src_user),
                                                                              replace=True)]
    src_other_features = src_sample_others[common_features]
    src_other_features.insert(loc = len(src_other_features.columns),column = 'User_type',value = 0)

    src_domain =  pd.concat([src_features, src_other_features], ignore_index=True)

    tar_user = target_data[target_data['User']==user_id]
    tar_features = tar_user[common_features]
    tar_features.insert(loc = len(tar_features.columns),column = 'User_type',value = 1)

    tar_other_users = target_data[target_data['User']!=user_id]
    tar_sample_others = tar_other_users.loc[np.random.RandomState(seed=42).choice(tar_other_users.index, size=len(tar_user), 
                                                                              replace=True)]
    tar_other_features = tar_sample_others[common_features]
    tar_other_features.insert(loc = len(tar_other_features.columns),column = 'User_type',value = 0)

    target_domain =  pd.concat([tar_features, tar_other_features], ignore_index=True)

    target_train, target_test = train_test_split(target_domain, test_size=0.8, random_state=42, 
                                             stratify=target_domain['User_type'])

    src_domain = src_domain.sample(frac = 1, random_state=42)
    target_train = target_train.sample(frac=1, random_state=42)
    target_test = target_test.sample(frac=1, random_state=42)


    r_source_domain = src_domain.to_numpy()
    r_target_train_domain = target_train.to_numpy()
    r_target_test_domain = target_test.to_numpy()

    z_src = r_source_domain[:, :]
    y_src = r_source_domain[:, -1]
        
    z_tar = r_target_train_domain[:, :]
    y_tar = r_target_train_domain[:, -1]

    x_tar1 = r_target_test_domain[:, :-1]
    y_tar1 = r_target_test_domain[:, -1]

    pos_label = np.ones((len(x_tar1),1))
    neg_label = np.zeros((len(x_tar1),1))

    z_tar_pos = np.hstack((x_tar1, pos_label))
    z_tar_neg = np.hstack((x_tar1, neg_label))

    z_tar1 = np.vstack((z_tar_pos,z_tar_neg))

    z_src = zscore(z_src, axis=0)
    z_tar = zscore(z_tar, axis=0)
    z_tar1 = zscore(z_tar1, axis=0)

    X_src_new, X_tar_new = MyTCA(z_src, z_tar, 'linear', 1, 1, dim+1)

    x_train_new = np.vstack((X_src_new,X_tar_new))
    y_train_new = np.append(y_src,y_tar)

    model = jdrf(x_train_new,y_train_new)

    X_trans = np.vstack((X_src_new,X_tar_new))
    Z_old = np.vstack((z_src,z_tar))

    TransZ = np.dot(Z_old.T, Z_old)
    inverZ = np.linalg.inv(TransZ)
    B = np.dot(np.dot(Z_old.T, X_trans), inverZ)

    Xt_new = z_tar1 @ B  

    predict_prob1 = model.predict_proba(Xt_new)
    pred_prob_pos = predict_prob1[:len(x_tar1),:]
    pred_prob_neg = predict_prob1[len(x_tar1):,:]

    cls_list = []
    for i in range (len(pred_prob_pos)):
        pos_0 = pred_prob_pos[i,0]
        pos_1 = pred_prob_pos[i,1]
    
        neg_0 = pred_prob_neg[i,0]
        neg_1 = pred_prob_neg[i,1]

        if (pos_0<pos_1):
            if (neg_0<neg_1):
                cls = 1
            else:
                if pos_1>neg_0:
                    cls =1
                else:
                    cls=0
        else:
            if (neg_0>neg_1):
                cls= 0
            else:
                if pos_0>neg_1:
                    cls=0
                else:
                    cls=1
                
        cls_list.append(cls)  

    predict = np.array(cls_list)
#predict = model.predict(Xt_new)

    acc = accuracy_score(y_tar1, predict)

    user_id_list.append (user_id)
    acc_list.append(acc)

df_final = pd.DataFrame(list(zip(user_id_list,acc_list)), columns = ['User', 'Accuracy'] )


#filename = f'Data_output/STL_same_device/User_' + str(user_id) +'.csv'
df_final.to_csv('STL_new.csv')
