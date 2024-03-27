import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(42)

def normalize(df):
    """
    The fumction to normalise source and target datasets. Use MinMax within the range of -1 and 1
    """
    scaler = MinMaxScaler(feature_range=(-1,1))
    model = scaler.fit(df)
    scaled = model.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    
    return scaled_df

def split_dataset(df,train_ratio, val_ratio):
    """
    Plit the dataset into 3 sections. Initially select samples with train ratio and get the train set. 
    From the remaining samples select validation samples using the given val_ratio.
    The remaining samples are in the test set.
    """
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


class TransferEncoder(object):
    def __init__(self, n_steps):
        """
        The transfer encoder learns a mapping between source domain and target domain samples.
        """
        self.n_steps = n_steps
        
    
    def fit_layers(self, X_train, Z_train, X_val, Z_val, batch_size = float('inf')):
        # Give the range of laters from 1-1000 with the gap of 50 to find the optimum no of hiddewn layers for each user
        hidden_layers_range = range(1, 1001, 50)  
        layers_list = []
        epoch_list = []
        loss_list = []

        # check it for each hidden layer values (ex: 50,100,150.....)
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
                y = tf.nn.tanh(tf.matmul(x, W) + b1)  
                z_hat = tf.nn.tanh(tf.matmul(y, W2) + b2)
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
                return cost
            
            def inference(X, Z):
                """
                Define this new function to do the inference using validation set. 
                """
                if X.ndim == 1:
                    X = X[np.newaxis, :]
                trans_X = forward(X)
                error = tf.reduce_sum(tf.square(Z - trans_X), axis=1)
                cost = tf.reduce_mean(tf.sqrt(error))
    
                return cost

            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

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
                
                train_cost = train_step(batch_xs, batch_zs)
                val_cost = inference(batch_xs_val,batch_zs_val)
                
                if val_cost < best_loss: # Using the validation loss, trying to find the best no of hidden layers and best epochs.
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

        # Get the layer id, epoach id and loss to the dataframe one the for loop complete.    
        df = pd.DataFrame(list(zip(layers_list,epoch_list,loss_list)), columns=['Layers', 'Epochs','Loss'])

        # Selecting the best no of hodden layers and epochs using the minimum loss.
        best_num_layers = int(df[df['Loss']==min(df['Loss'])]['Layers'])
        best_epochs = int(df[df['Loss']==min(df['Loss'])]['Epochs'])
        
        print(f'The optimum number of layers is {best_num_layers} and the optimum number of epochs is {best_epochs+1}')

        input_dim = X_train[0].shape[0]

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        tf.random.set_seed(1234)
        W = tf.Variable(tf.random.uniform([input_dim, best_num_layers], -1.0 / np.sqrt(input_dim), 1.0 / np.sqrt(input_dim), 
                                          seed=1234))

        # Initialize b to zero
        b1 = tf.Variable(tf.zeros([best_num_layers]))

        # Use tied weights
        W2 = tf.transpose(W)
        b2 = tf.Variable(tf.zeros([input_dim]))
    
        @tf.function
        def forward(x):
            y = tf.nn.tanh(tf.matmul(x, W) + b1)  
            z_hat = tf.nn.tanh(tf.matmul(y, W2) + b2)
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
            return cost
        
        def inference(X, Z):
            if X.ndim == 1:
                X = X[np.newaxis, :]
            trans_X = forward(X)
            error = tf.reduce_sum(tf.square(Z - trans_X), axis=1)
            cost = tf.reduce_mean(tf.sqrt(error))
    
            return cost

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

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
        
        for epoch in range(best_epochs+1):
            start = 0
            end_train = start + batch_size
            end_val = start + batch_size_val
            
            batch_xs = X[start:end_train]
            batch_zs = Z[start:end_train]
            
            batch_xs_val = X_val[start:end_val]
            batch_zs_val = Z_val[start:end_val]
            
            train_cost = train_step(batch_xs, batch_zs)
            val_cost = inference(batch_xs_val,batch_zs_val)
            
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
    
    te.fit_layers(inputs_train, outputs_train,inputs_val,outputs_val)
    return te


def jdrf(x_train, y_train,x_test, y_test):
    """
    Use Random Forest classifer with hyperparamter tuning to evalaute the binary classification
    """
    param_grid = {'bootstrap': [False, True],
              'max_depth': [30,40],
              'min_samples_leaf': [2,4],
              'min_samples_split': [5,6],
              'n_estimators': [500, 1000],
              'criterion':['gini']
             }

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_rf = GridSearchCV(RandomForestClassifier(), param_grid, refit = True,
                           cv = cv_inner, n_jobs = 1, scoring = 'accuracy')
    
    tuning = grid_rf.fit(x_train, y_train)

    best_hyperparams = tuning.best_params_

    best_model = RandomForestClassifier(random_state=42, **best_hyperparams)
#     best_model = RandomForestClassifier(random_state=42)

    trained_model = best_model.fit(x_train, y_train)
    pred_prob = trained_model.predict_proba(x_test)
    pred = trained_model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    return acc, pre, rec, f1, pred_prob


# Load the two datasets
source_data = pd.read_csv('C:/Users/s3929438/all_features_mobile_100_all_final_latest.csv', index_col=[1,2])
target_data = pd.read_csv('C:/Users/s3929438/all_features_tablet_100_all_latest.csv',index_col=[1,2])

source_data = source_data.loc[:, ~source_data.columns.str.contains('^Unnamed')]
target_data = target_data.loc[:, ~target_data.columns.str.contains('^Unnamed')]

# conver the values of dataframes to float32 data type.
source_data = source_data.astype('float32')
target_data = target_data.astype('float32')

# Define the common features need to be selected from each dataframe.
common_features = ['mean_F1', 'mean_F2', 'mean_F3', 'mean_F4', 'Tri_graph', 'Error_rate_%','mean_hold_time',
                    'mean_F1_dis_1_LL','mean_F1_dis_2_LL', 'mean_F1_dis_2_RR', 'mean_F1_dis_3_LL','mean_F1_dis_3_RR', 
                    'mean_F2_dis_1_LL','mean_F2_dis_2_LL', 'mean_F2_dis_2_RR', 'mean_F2_dis_3_LL','mean_F2_dis_3_RR', 
                    'mean_F3_dis_2_RR', 'mean_F3_dis_3_LL','mean_F4_dis_1_LL','mean_F4_dis_2_LL', 'mean_F4_dis_2_RR',
                    'mean_F4_dis_3_LL','mean_F1_se', 'mean_F2_se', 'mean_F3_se','mean_F4_se', 'mean_F1_th', 'mean_F2_th', 
                    'mean_F3_th', 'mean_F4_th']

np.random.seed(42)

BBMAS_TE_FACTORY = lambda: TransferEncoder(n_steps=2000)
te_factory = BBMAS_TE_FACTORY
te = te_factory()


i = 1 # User ID

# Select the samples of the define User ID from Source dataset
src_user = source_data.loc[source_data.index.get_level_values(0)==i]
src_user = src_user[common_features]

# Select the samples of the define User ID from Target dataset
tar_user = target_data.loc[target_data.index.get_level_values(0)==i]
tar_user = tar_user[common_features]

# Rearrange the columns of target dataset as same as the source dataset
col_order = src_user.columns
tar_user = tar_user.reindex(columns=col_order)

# Normalise the source and target datasets
indexes_src = src_user.index
src_user, _ = normalize(src_user)
src_user.index = indexes_src

indexes_tar = tar_user.index
tar_user, _ = normalize(tar_user)
tar_user.index = indexes_tar

# Split each dataset into train, validation and test.
src_train, src_val, src_test = split_dataset(src_user, 0.3, 0.2)
tar_train, tar_val, tar_test = split_dataset(tar_user, 0.3, 0.2)

# Activate the autoencoder and train it
te = fit_te(te, src_train, tar_train, src_val,tar_val, one_to_one=False)

# Get the original test set of Source dataset.
source_new = src_test

# transfer the loaded source_new dataset using the trained autoencoder.
source_new.values[:] = te.transfer(source_new.values)

# Insert a Class label column as genuine user.
source_new.insert(loc = len(source_new.columns),column = 'User_type',value = 1)

# Selecting same no of samples from all other users from the target dataset only, normalise it and label them as intruders.
tar_all_imposters = target_data.loc[target_data.index.get_level_values(0)!=i]
tar_all_imposters = tar_all_imposters[common_features]
col_order = src_user.columns
tar_user = tar_all_imposters.reindex(columns=col_order)
indexes_tar = tar_all_imposters.index
tar_all_imposters, _ = normalize(tar_all_imposters)
tar_all_imposters.index = indexes_tar
tar_all_imposters.insert(loc = len(tar_all_imposters.columns),column = 'User_type',value = 0)
tar_all_imposters.index = tar_all_imposters.index.map(lambda x: ','.join(map(str, x)))
imposter_sample_index = list(np.random.choice(tar_all_imposters.index.get_level_values(0), size=len(source_new), replace=False))
other_imposters_index = list(tar_all_imposters.index.difference(imposter_sample_index))
tar_imposters = tar_all_imposters.loc[tar_all_imposters.index.get_level_values(0).isin(imposter_sample_index)]

# Combining the genuine and imposter samples created the train set.
train_set =  pd.concat([source_new, tar_imposters], ignore_index=True)
train_set = train_set.sample(frac = 1, random_state=42)

# Getting the target test set as our testing samples.
tar_user_new = tar_test
tar_user_new.insert(loc = len(tar_user_new.columns),column = 'User_type',value = 1)

# Selecting same number of samples from target dataset and 
tar_all_imposters_new = tar_all_imposters.loc[tar_all_imposters.index.get_level_values(0).isin(other_imposters_index)]
tar_imposter_new = tar_all_imposters_new.loc[np.random.RandomState(seed=42).choice(tar_all_imposters_new.index,size=len(tar_user_new), 
                                                                                   replace=False)]
# Combining the two and creating the test set.    
test_set = pd.concat([tar_user_new, tar_imposter_new], ignore_index=True)
test_set = test_set.sample(frac = 1, random_state=42)

# Defining x and y of train and test sets.
train_x = train_set.iloc[:,:-1]
train_y = train_set.iloc[:,-1]
test_x = test_set.iloc[:,:-1]
test_y = test_set.iloc[:,-1]

# Binary classification using Random Forest.
jdrf(train_x,train_y,test_x, test_y)