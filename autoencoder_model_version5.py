import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
from statistics import mean, stdev
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
np.random.seed(42)

class TransferEncoder(object):
    def __init__(self, n_steps):
        """
        The transfer encoder learns a mapping between source domain and target domain samples.
        """
#         self.n_hidden = n_hidden
        self.n_steps = n_steps
        
    
    def fit_layers(self, X_train, Z_train, X_val, Z_val, batch_size = float('inf')):
        hidden_layers_range = range(50, self.n_steps+1, 50)
        layers_list = []
        epoch_list = []
        loss_list = []

        for n_hidden in hidden_layers_range:
            input_dim = X_train[0].shape[0]

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
            tf.random.set_seed(1234)
            W = tf.Variable(tf.random.uniform([input_dim, n_hidden], -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim), 
                                          seed=1234))

        # Initialize b to zero
            b1 = tf.Variable(tf.zeros([n_hidden]))

        # Use tied weights
            W2 = tf.transpose(W)
            b2 = tf.Variable(tf.zeros([input_dim]))
    
            @tf.function
            def forward(x):
                y = tf.nn.sigmoid(tf.matmul(x, W) + b1)  
                z_hat = tf.nn.sigmoid(tf.matmul(y, W2) + b2)
                return z_hat

            @tf.function
            def train_step(x_batch, z_batch):
                with tf.GradientTape() as tape:
                    z_hat = forward(x_batch)
                    error = tf.reduce_sum(tf.square(z_batch - z_hat), axis=1)
                    cost = tf.reduce_mean(tf.sqrt(error))
                # Add L2 regularization
                    l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in [W, b1, b2]])
                    cost += l2_loss
                gradients = tape.gradient(cost, [W, b1, b2])
                optimizer.apply_gradients(zip(gradients, [W, b1, b2]))
                return cost, W, b1, b2
            
            def inference(X, Z, W, b1, b2):
                y = tf.nn.sigmoid(tf.matmul(X, W) + b1)
                W2 = tf.transpose(W)
                trans_X = tf.nn.sigmoid(tf.matmul(y, W2) + b2)
                error = tf.reduce_sum(tf.square(Z - trans_X), axis=1)
                cost = tf.reduce_mean(tf.sqrt(error))
    
                return cost

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            n_samples = len(X_train)
            idx = tf.random.shuffle(tf.range(n_samples))
            X = tf.gather(X_train, idx)
            Z = tf.gather(Z_train, idx)
            
            n_samples_val = len(X_val)
            idx_val = tf.random.shuffle(tf.range(n_samples_val))
            X_val = tf.gather(X_val, idx_val)
            Z_val = tf.gather(Z_val, idx_val)

            batch_size = min(n_samples, batch_size)
            batch_size_val = min(n_samples_val, batch_size)
            
            best_loss = float('inf')
            best_loss_val = 0
            no_of_layers = 0
            early_stopping_counter = 0
            best_epoch = 0
        
            for epoch in range(self.n_steps):
                start = 0
                end_train = start + batch_size
                end_val = start + batch_size_val
                batch_xs = X[start:end_train]
                batch_zs = Z[start:end_train]
                
                batch_xs_val = X_val[start:end_val]
                batch_zs_val = Z_val[start:end_val]
                
                train_cost, W_new, b1_new, b2_new = train_step(batch_xs, batch_zs)
                val_cost = inference(batch_xs_val,batch_zs_val, W_new, b1_new, b2_new)
                
                if (val_cost < best_loss):
                    best_loss = val_cost
                    early_stopping_counter = 0
                else:
                    early_stopping_counter +=1
                
                if early_stopping_counter >= 20:
                    no_of_layers = n_hidden
                    best_loss_val = val_cost.numpy()
                    best_epoch = epoch
                    print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
                    break
            
            layers_list.append(no_of_layers)
            epoch_list.append(best_epoch)
            loss_list.append(best_loss_val)
            
        df = pd.DataFrame(list(zip(layers_list,epoch_list,loss_list)), columns=['Layers', 'Epochs','Loss'])

        best_num_layers = int(df[df['Loss']==min(df['Loss'])]['Layers'])
        best_epochs = int(df[df['Loss']==min(df['Loss'])]['Epochs'])
        
        print(f'The optimum number of layers is {best_num_layers} and the optimum number of epochs is {best_epochs+1}')
        
        return best_num_layers, best_epochs
    
    def fit(self, X_train, Z_train, X_val, Z_val,best_layers, best_steps, batch_size = float('inf')):
        """
        X is the list of samples in the source domain and Z is the list of samples in the target domain. X and Z should
         have equal length. Model parameters are determined using gradient descent optimization with n_steps as
         specified in the constructor.
        """
        input_dim = X_train[0].shape[0]

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(tf.random.uniform([input_dim, best_layers], -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim), 
                                          seed=np.random.randint(0, 1e9)))

        # Initialize b to zero
        b1 = tf.Variable(tf.zeros([best_layers]))

        # Use tied weights
        W2 = tf.transpose(W)
        b2 = tf.Variable(tf.zeros([input_dim]))

        @tf.function
        def forward(x):
            y = tf.nn.sigmoid(tf.matmul(x, W) + b1)
            z_hat = tf.nn.sigmoid(tf.matmul(y, W2) + b2)
            return z_hat

        @tf.function
        def train_step(x_batch, z_batch):
            with tf.GradientTape() as tape:
                z_hat = forward(x_batch)
                error = tf.reduce_sum(tf.square(z_batch - z_hat), axis=1)
                cost = tf.reduce_mean(tf.sqrt(error))
            gradients = tape.gradient(cost, [W, b1, b2])
            optimizer.apply_gradients(zip(gradients, [W, b1, b2]))
            return cost, W, b1, b2
        
        def inference(X, Z, W, b1, b2):
            y = tf.nn.sigmoid(tf.matmul(X, W) + b1)
            W2 = tf.transpose(W)
            trans_X = tf.nn.sigmoid(tf.matmul(y, W2) + b2)
            error = tf.reduce_sum(tf.square(Z - trans_X), axis=1)
            cost = tf.reduce_mean(tf.sqrt(error))
    
            return cost

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        n_samples = len(X_train)
        idx = tf.random.shuffle(tf.range(n_samples))
        X = tf.gather(X_train, idx)
        Z = tf.gather(Z_train, idx)
        
        n_samples_val = len(X_val)
        idx_val = tf.random.shuffle(tf.range(n_samples_val))
        X_val = tf.gather(X_val, idx_val)
        Z_val = tf.gather(Z_val, idx_val)

        batch_size = min(n_samples, batch_size)
        batch_size_val = min(n_samples_val, batch_size)

        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(best_steps+1):
            start = 0
            end_train = start + batch_size
            end_val = start + batch_size_val
            
            batch_xs = X[start:end_train]
            batch_zs = Z[start:end_train]
            
            batch_xs_val = X_val[start:end_val]
            batch_zs_val = Z_val[start:end_val]
            
            train_cost, W_new, b1_new, b2_new = train_step(batch_xs, batch_zs)
            val_cost = inference(batch_xs_val,batch_zs_val, W_new, b1_new, b2_new)
            
            train_loss_history.append(train_cost)
            val_loss_history.append(val_cost)


        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.show()

        self.forward = forward
        return 

    def transfer(self, X):
        """
        Reconstruct X in the target domain.
        """
        if X.ndim == 1:
            X = X[np.newaxis, :]
        return self.forward(X)  

def fit_te(te, df_source_train, df_target_train,df_source_val, df_target_val, one_to_one):
    """
    Fit the ITE using the given source and target domain dataframes and bipartite strategy
    """
    inputs_train, outputs_train = [], []
    inputs_val, outputs_val = [],[]
    
    user_pairs = [(u, u) for u in df_source_train.index.get_level_values(0).unique()]

    if one_to_one:
        iter_fun = zip
    else:
        iter_fun = product

    for u1, u2 in user_pairs:
        idx1 = np.arange(df_source_train.loc[u1].shape[0])
        idx2 = np.arange(df_target_train.loc[u2].shape[0])
        idx = np.array(list(iter_fun(idx1, idx2)))
        inputs_train.append(df_source_train.loc[u1].values[idx[:, 0]])
        outputs_train.append(df_target_train.loc[u2].values[idx[:, 1]])

    inputs_train = np.concatenate(inputs_train)
    outputs_train = np.concatenate(outputs_train)
    
    for u1, u2 in user_pairs:
        idx1_val = np.arange(df_source_val.loc[u1].shape[0])
        idx2_val = np.arange(df_target_val.loc[u2].shape[0])
        idx_val = np.array(list(iter_fun(idx1_val, idx2_val)))
        inputs_val.append(df_source_val.loc[u1].values[idx_val[:, 0]])
        outputs_val.append(df_target_val.loc[u2].values[idx_val[:, 1]])
    
    inputs_val = np.concatenate(inputs_val)
    outputs_val = np.concatenate(outputs_val)
    
    best_layers, best_steps  = te.fit_layers(inputs_train, outputs_train,inputs_val,outputs_val)
    te.fit(inputs_train, outputs_train,inputs_val,outputs_val, best_layers, best_steps)
    return te

def normalize(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    
    return scaled_df, scaler

def split_dataset(df,train_ratio, val_ratio):
    np.random.seed(42)
    sample_index = list(np.random.choice(df.index.get_level_values(1), int(round((len(df) * train_ratio), 0)), 
                                         replace=False))
    other_index = list(df.index.get_level_values(1).difference(sample_index))
    val_index = list(np.random.choice(other_index, int(round((len(other_index) * val_ratio), 0)), replace=False))
    union_indices = set(sample_index).union(val_index)
    test_index = list(df.index.get_level_values(1).difference(union_indices))
    train = df.loc[df.index.get_level_values(1).isin(sample_index)]
    val = df.loc[df.index.get_level_values(1).isin(val_index)]
    test = df.loc[df.index.get_level_values(1).isin(test_index)]
    
    return train, val, test

def jdrf(x_train, y_train,x_test, y_test):

    best_model = RandomForestClassifier(random_state=42)

    trained_model = best_model.fit(x_train, y_train)
    pred_prob = trained_model.predict_proba(x_test)
    pred = trained_model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    return acc, pre, rec, f1, pred_prob

def transfer_learning(user_id,source_data,target_data):
    np.random.seed(42)
    BBMAS_TE_FACTORY = lambda: TransferEncoder(n_steps=2000)
    te_factory = BBMAS_TE_FACTORY
    te = te_factory()

    i = user_id
    src_user = source_data.loc[source_data.index.get_level_values(0)==i]
    tar_user = target_data.loc[target_data.index.get_level_values(0)==i]
    col_order = src_user.columns
    tar_user = tar_user.reindex(columns=col_order)
        
    src_train, src_val, src_test = split_dataset(src_user, 0.3, 0.2)
    tar_train, tar_val, tar_test = split_dataset(tar_user, 0.3, 0.2)
    
    indexes_src_tarin = src_train.index
    src_train, src_scaler = normalize(src_train)
    src_train.index = indexes_src_tarin
    indexes_src_val = src_val.index
    val_scale = src_scaler.transform(src_val)
    src_val = pd.DataFrame(val_scale, columns=src_val.columns)
    src_val.index = indexes_src_val
    
    indexes_tar_tarin = tar_train.index
    tar_train,tar_scaler = normalize(tar_train)
    tar_train.index = indexes_tar_tarin
    indexes_tar_val = tar_val.index
    val_scale_tar = tar_scaler.transform(tar_val)
    tar_val = pd.DataFrame(val_scale_tar, columns=tar_val.columns)
    tar_val.index = indexes_tar_val

    te = fit_te(te, src_train, tar_train, src_val,tar_val, one_to_one=False)
    
    test_scale = src_scaler.transform(src_test)
    src_test = pd.DataFrame(test_scale, columns=src_test.columns)

    source_new = src_test

    source_new.values[:] = te.transfer(source_new.values)
    source_new.insert(loc = len(source_new.columns),column = 'User_type',value = 1)
    
    return source_new, col_order


def user_training(user_id, target_data,tranfer_set, col_order):
    tar_user = target_data.loc[target_data.index.get_level_values(0)==user_id]
    tar_user = tar_user[common_features]
    tar_user = tar_user.reindex(columns=col_order)
    tar_user.insert(loc = len(tar_user.columns),column = 'User_type',value = 1)
    tar_train, tar_val, tar_test = split_dataset(tar_user, 0.3, 0.2)
    
    tar_all_imposters = target_data.loc[target_data.index.get_level_values(0)!=user_id]
    tar_all_imposters = tar_all_imposters[common_features]
    tar_all_imposters = tar_all_imposters.reindex(columns=col_order)
    tar_all_imposters.insert(loc = len(tar_all_imposters.columns),column = 'User_type',value = 0)
    tar_all_imposters.index = tar_all_imposters.index.map(lambda x: ','.join(map(str, x)))
    imposter_set_size = len(tar_train) + len(tar_val) + len(tranfer_set)
    imposter_sample_index = list(np.random.choice(tar_all_imposters.index.get_level_values(0), size=imposter_set_size, 
                                                  replace=False))
    other_imposters_index = list(tar_all_imposters.index.difference(imposter_sample_index))
    tar_imposter_train = tar_all_imposters.loc[tar_all_imposters.index.get_level_values(0).isin(imposter_sample_index)]

    train_set_target =  pd.concat([tar_train, tar_val,tar_imposter_train], ignore_index=True)
    
    x_train_tar = train_set_target.iloc[:,:-1]
    y_tarin_tar = train_set_target.iloc[:,-1]
    
    norm_x_train_tar, scaler = normalize(x_train_tar)
    
    x_train_src = tranfer_set.iloc[:,:-1]
    y_train_src = tranfer_set.iloc[:,-1]
    
    train_x =  pd.concat([x_train_src, norm_x_train_tar], ignore_index=True)
    train_y =  pd.concat([y_train_src, y_tarin_tar], ignore_index=True)
    
    print(train_x.shape, train_y.shape)
    
    tar_all_imposters_test = tar_all_imposters.loc[tar_all_imposters.index.get_level_values(0).isin(other_imposters_index)]
    tar_imposter_test = tar_all_imposters_test.loc[np.random.RandomState(seed=42).choice(tar_all_imposters_test.index.get_level_values(0),size=len(tar_test),replace=False)]
    
    test_set_target = pd.concat([tar_test, tar_imposter_test], ignore_index=True)

    test_x = test_set_target.iloc[:,:-1]
    test_y = test_set_target.iloc[:,-1]

    scaled = scaler.transform(test_x)
    norm_test_x = pd.DataFrame(scaled, columns=test_x.columns)

    acc, pre, rec, f1, pred_prob = jdrf(train_x,train_y,norm_test_x, test_y)
    
    return acc, pre, rec, f1, pred_prob


source_data = pd.read_csv('C:/Users/s3929438/all_features_mobile_100_all_final_latest.csv', index_col=[1,2])
target_data = pd.read_csv('C:/Users/s3929438/all_features_tablet_100_all_latest.csv',index_col=[1,2])
# all_features_tablet_100_all_latest

source_data = source_data.loc[:, ~source_data.columns.str.contains('^Unnamed')]
target_data = target_data.loc[:, ~target_data.columns.str.contains('^Unnamed')]

source_data = source_data.astype('float32')
target_data = target_data.astype('float32')

common_features = ['mean_F1', 'mean_F2', 'mean_F3', 'mean_F4', 'Tri_graph', 'Error_rate_%','mean_hold_time',
                    'mean_F1_dis_1_LL','mean_F1_dis_2_LL', 'mean_F1_dis_2_RR', 'mean_F1_dis_3_LL','mean_F1_dis_3_RR', 
                    'mean_F2_dis_1_LL','mean_F2_dis_2_LL', 'mean_F2_dis_2_RR', 'mean_F2_dis_3_LL','mean_F2_dis_3_RR', 
                    'mean_F3_dis_2_RR', 'mean_F3_dis_3_LL','mean_F4_dis_1_LL','mean_F4_dis_2_LL', 'mean_F4_dis_2_RR',
                    'mean_F4_dis_3_LL']


source_data = source_data[common_features]
target_data = target_data[common_features]

source_new, col_order = transfer_learning(1,source_data, target_data)
user_training(1, target_data,source_new,col_order)