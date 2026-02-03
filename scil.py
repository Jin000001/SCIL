# -*- coding: utf-8 -*-
import os
from sklearn.metrics import f1_score
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from collections import deque
from aux_loss_functions import  calc_reconstruction_loss_vec,calc_kl_loss_vec
from scipy.stats import ranksums,mannwhitneyu
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind
from sklearn.neighbors import KernelDensity
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras import backend as K
from scipy.stats import ks_2samp
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.optimizers import Adam
from scipy.spatial.distance import pdist, squareform
from ae_mlp import train_ae_mlp, predict_with_ae_mlp,predict_with_ae_mlp_noembed
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy
from ae_mlp import set_seed
from data.con2.sea_new_con import load_sea_ref
from data.con2.sine_con2 import load_sine_ref
from data.con2.vib_con2 import load_vib_ref
from data.con2.forest_con2 import load_forest_ref
from data.sensorless.sensorless_dataprepapre import load_sensor_ref
from data.shuttle.shuttle_load import load_shuttle_ref
from data.con2.kdd import load_kdd_ref
from data.con2.mnist01_con2 import load_mnist_ref
from data.con2.blob import load_blob_ref
from data.water_network.waterNet import load_water_ref
from itertools import combinations
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.cluster import DBSCAN
import time
from sklearn.covariance import EmpiricalCovariance, MinCovDet, LedoitWolf, OAS

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

##########################
# Prequential evaluation #
##########################


def plot_input_and_embedding(X_train, embeddings, y_train_up):
    # Step 1: Apply t-SNE to reduce X_train to 2D
    tsne_params = {
        "n_components": 2,
        "perplexity": 30,             # ↓ Reduce neighborhood perception
        "learning_rate": 500,         # ↑ Increase learning rate
        "n_iter": 300,               # ↑ More iterations
        "init": 'random',
        "random_state": 42
    }

    # t-SNE on input
    X_train_2d = TSNE(**tsne_params).fit_transform(X_train)

    # t-SNE on embeddings
    embeddings_2d = TSNE(**tsne_params).fit_transform(embeddings)

    # Step 3: Plot the input data in 2D
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_up, cmap='viridis', alpha=0.7)
    plt.title('Input Data (t-SNE)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    # Step 4: Plot the embedding data in 2D
    plt.subplot(1, 2, 2)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_train_up, cmap='viridis', alpha=0.7)
    plt.title('Embedding Space (t-SNE)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    plt.show()



def lookahead_predictions_iforest(t, d_env, iforest):
    lookahead_examples = d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]

    lookahead_pred_class = iforest.predict(lookahead_examples)
    lookahead_pred_class[lookahead_pred_class == 1] = 0
    lookahead_pred_class[lookahead_pred_class == -1] = 1

    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class

def lookahead_predictions_lof(t, d_env, lof):
    lookahead_examples = d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]

    lookahead_pred_class = lof.predict(lookahead_examples)
    lookahead_pred_class[lookahead_pred_class == 1] = 0
    lookahead_pred_class[lookahead_pred_class == -1] = 1

    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class


def create_iforest(d_env):
    #default,max_samples=100
    return IsolationForest(max_samples='auto', contamination='auto', random_state=d_env['random_state'])



def create_lof(d_env):
    return LocalOutlierFactor(n_neighbors=3,novelty=True)

def train_iforest(q, iforest):
    # convert queue to array
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    arr_unlabelled = np.concatenate(q_lst, axis=0)

    # train
    iforest.fit(arr_unlabelled)
def train_lof(q, lof):
    # convert queue to array
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    arr_unlabelled = np.concatenate(q_lst, axis=0)

    # train
    lof.fit(arr_unlabelled)

def are_nearly_continuous(timesteps):
    n = len(timesteps)
    count_nearly_continuous=0
    # Check if there are at least 30 timesteps
    if n < 30:
        return False

    # Get the last 100 timesteps
    last_100 = timesteps[-30:]
    window = last_100[0:30]
    if max(window) - min(window) <= 70:
        count_nearly_continuous += 1
    return count_nearly_continuous > 0

def load_arr(d):
    t_drift = [5000, 10000]
    if 'water' in params_env['data_source']:
        df=pd.read_csv('water_mulincr_arr.csv')
        t_drift = [4000]

    if 'kdd' in params_env['data_source']:
        df = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\kdd_multiclass_novel_ori_onehot.csv')
    if 'vicon' in params_env['data_source']:
        # df = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\vib_multiclass_seq_point_rep_2drift_unscaled_novel_ori.csv')
        #2percent
        # df = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\vib_gradual_2drift.csv')
        # 5percent
        # df = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\vib_gradual_2drift_5percent.csv')
        # 10percent
        df = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\vib_gradual_2drift_10percent.csv')
    if 'blob' in params_env['data_source']:
        # df = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\blob_mulincr_arr.csv')
        #2percent
        df= pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\blobs_6class_arr_drift.csv')
        # 5percent
        # df= pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\blobs_6class_arr_drift_5percent.csv')
        # 10percent
        # df= pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\blobs_6class_arr_drift_10percent.csv')

    if 'forescon' in params_env['data_source']:
        # df = pd.read_csv(r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\forest_multiclass_seq_point_drift_novel_ori.csv")
        #2percent
        df = pd.read_csv(r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\forest_gradual_2drift.csv")
        #5percent
        # df = pd.read_csv(
        #     r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\forest_gradual_2drift_5percent.csv")
        #10percent
        # df = pd.read_csv(
        #     r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\forest_gradual_2drift_10percent.csv")


    if 'shuttle' in params_env['data_source']:
        df = pd.read_csv(
            r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\shuttle\shuttle_online.csv")
        t_drift = [100000]
    if 'sea_con' in params_env['data_source']:
        df = pd.read_csv(r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\sea_mulincr_arr.csv")
    if 'mniscon' in params_env['data_source']:
        # df = pd.read_csv(r"C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\mnist01_multiclass_seq_point_novel_ori.csv")
        df=pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\mnist01_multiclass_allclass.csv')
        t_drift = [2500]
    if 'sensorless' in params_env['data_source']:
        #2percent
        # df=pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\sensorless\sensorless_arr_2percent.csv')

        # #4 percent
        df=pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\sensorless\sensorless_arr.csv')
        #
        # #10 percent
        # df=pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\sensorless\sensorless_arr_10percent.csv')

        t_drift = [2500]
    data_init_unlabelled = df.iloc[:2000, :].drop('class', axis=1).to_numpy()

    df=df.values
    return df, data_init_unlabelled, t_drift

def pretrain_aemlp_mulincr(d):
    set_seed(42)
    if 'vicon' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\vib_offline_clf.csv')
        input_dim = 10  # Input dimension
        latent_dim = 2
        A = 40
        B = 1

    if 'blob' in params_env['data_source']:
        # df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\blob_offline_clf.csv')
        df_train=pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\blobs_6class_offline.csv')
        input_dim = 3  # Input dimension
        latent_dim = 2
        A = 300
        B = 5
    if 'forescon' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\forest_offline_clf.csv')#
        ###plot input and embed
        # df_train = pd.read_csv('sampled_shuffled_data.csv')
        input_dim = 52 # Input dimension
        latent_dim = 20
        A = 0.1
        B = 1
    if 'sensorless' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\sensorless\sensorless_offline.csv')#
        input_dim = 48 # Input dimension
        latent_dim = 20
        A = 0.1
        B = 1
    if 'shuttle' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\shuttle\shuttle_offline.csv')#
        input_dim = 9 # Input dimension
        latent_dim = 2
        A = 200000
        B = 1

    if 'sea_con' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\sea_offline_clf.csv')
        input_dim = 2 # Input dimension
        latent_dim = 2
        A = 20
        B = 2
    if 'mniscon' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\mnist_offline_clf.csv')
        input_dim = 784 # Input dimension
        latent_dim = 20
        A = 0.1
        B = 1
    if 'kdd' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\con2\kdd_offline_clf.csv')
        input_dim = 116  # Input dimension
        latent_dim = 20
        A = 0.3
        B = 1
    if 'water' in params_env['data_source']:
        df_train = pd.read_csv(r'C:\Users\jli00001\OneDrive - University of Cyprus\Desktop\online learning\2nd conference papar\VAE4AS\data\water_network\water_conta_ref.csv')
        input_dim = 1  # Input dimension
        latent_dim = 2
        A = 10
        B = 1

    X_train = df_train.drop('class', axis=1).to_numpy()
    y_train = df_train['class'].to_numpy()
##upsampling here: sampling with replacement (Increase sample count by replicating existing samples)
    arr=X_train
    majority_class = arr[:1000]
    minority_class_1 = arr[2000:2030]

#####plot input and embed
    X_train_up=X_train
    y_train_up=y_train
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    # Create Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    class DiffusionModel(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(DiffusionModel, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8),
                nn.ReLU(),
                nn.Linear(8, input_dim),
                nn.Sigmoid()  # Ensure output in [0, 1] range
            )

        def forward(self, x):
            z = self.encoder(x)
            recon_x = self.decoder(z)
            return recon_x, z

    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            return x

    if 'water' in params_env['data_source']:
        class DiffusionModel(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(DiffusionModel, self).__init__()
                self.encoder = nn.Sequential(

                    nn.Linear(input_dim, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, input_dim),
                    nn.Sigmoid()  # Ensure output in [0, 1] range
                )

            def forward(self, x):
                z = self.encoder(x)
                recon_x = self.decoder(z)
                return recon_x, z

        class MLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, num_classes)

            def forward(self, x):
                x = self.fc1(x)
                return x

    # Model parameters
    if 'kdd' in params_env['data_source']:
        class DiffusionModel(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(DiffusionModel, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                recon_x = self.decoder(z)
                return recon_x, z

        class MLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, num_classes)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    if 'forescon' in params_env['data_source'] or 'sensorless' in params_env['data_source']:
        class DiffusionModel(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(DiffusionModel, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                recon_x = self.decoder(z)
                return recon_x, z

        class MLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, num_classes)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    if 'mniscon' in params_env['data_source']:
        class DiffusionModel(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(DiffusionModel, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                recon_x = self.decoder(z)
                return recon_x, z

        class MLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, num_classes)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x



    num_classes = len(np.unique(y_train))


    # Instantiate models
    if params_env['ae']=='noembed':
        diffusion_model = DiffusionModel(input_dim, latent_dim)
        mlp = MLP(input_dim, num_classes)
    else:
        diffusion_model = DiffusionModel(input_dim, latent_dim)
        mlp = MLP(latent_dim, num_classes)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(list(diffusion_model.parameters()) + list(mlp.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train model
    num_epochs = 20




    for epoch in range(num_epochs):
        diffusion_model.train()
        mlp.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_loss = []

        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            # data = (data - data.min()) / (data.max() - data.min())

            # Diffusion model forward pass
            recon_batch, z = diffusion_model(data)

            # Classifier forward pass
            if params_env['ae'] == 'noembed':
                output = mlp(data)
            else:
                output = mlp(z)

            # Calculate loss
            if 'kdd' in params_env['data_source'] or 'vicon' in params_env['data_source'] or 'forescon' in params_env['data_source'] or 'blob' in params_env['data_source'] or 'water' in params_env['data_source'] or 'mniscon' in params_env['data_source'] or 'sea_con' in params_env['data_source'] in params_env['data_source'] or 'sensorless' in params_env['data_source'] or 'shuttle' in params_env['data_source']:
                recon_loss = F.mse_loss(recon_batch, data)

            clf_loss = criterion(output, target)


            recon_loss = recon_loss / A
            clf_loss = clf_loss / B
            # print('recon loss',recon_loss)
            # print('clf loss',clf_loss)

            if params_env['com']== '28':
                loss = 0.2*recon_loss + 0.8*clf_loss
            elif params_env['com']== '55':
                loss = 0.5*recon_loss + 0.5*clf_loss
            elif params_env['com']== '82':
                loss = 0.8 * recon_loss+ 0.2 * clf_loss

            # loss = 0.2*recon_loss + 0.8*clf_loss
            # loss = 0.5 * recon_loss + 0.5 * clf_loss
            # loss = 0.8 * recon_loss + 0.2 * clf_loss


            # Backward propagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        print(
            f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}, Accuracy: {100. * correct / total:.2f}%')


    # Make predictions on entire dataset and get outputs
    diffusion_model.eval()
    mlp.eval()
    all_outputs = []
    all_recondata = []
    embed_all=[]

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            recon_batch, z = diffusion_model(data)
            if params_env['ae']=='noembed':
                output = mlp(data)
            else:
                output = mlp(z)
            embed_all.append(z)
            all_outputs.append(output)
            all_recondata.append(recon_batch)


#####plot input and embedding
    embed_all_stacked = torch.cat(embed_all, dim=0).cpu().numpy()
    print(np.shape(X_train_up),np.shape( embed_all_stacked),np.shape(y_train_up))
    # plot_input_and_embedding(X_train_up,  embed_all_stacked,y_train_up)
#####
    output = torch.cat(all_outputs, dim=0)
    recondata = torch.cat(all_recondata, dim=0)




    ref_logit0 = output[:1000]
    ref_logit1 = output[1000:1030]

    ae_loss = torch.mean((X_train - recondata) ** 2, dim=1)


    ae_loss = ae_loss.cpu().detach().numpy()

    loss_0_recon = ae_loss[:1000]
    class_0 =np.max(np.array(loss_0_recon))
    if len(ae_loss)>1000:
        loss_1_recon = ae_loss[1000:1030]
        class_1 = np.max(np.array(loss_1_recon))
    else:
        class_1=None


    if 'mniscon' in params_env['data_source']:
        class_0 = np.median(np.array(loss_0_recon))-0.5*np.std(np.array(loss_0_recon))
        class_1 = np.median(np.array(loss_1_recon))

    recon_ref = loss_0_recon[:50]
    print(class_0)




    return diffusion_model, mlp, output.cpu().detach().numpy(), ref_logit0.cpu().detach().numpy(), ref_logit1.cpu().detach().numpy(), class_0, class_1, recon_ref



def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric

###############
# Autoencoder #
###############


def retrain_aemlp(ae_model, mlp, X_train, y_train, num_epochs, lr, params_env):
    if 'vicon' in params_env['data_source']:
        A = 40
        B = 1

    if 'blob' in params_env['data_source']:
        A = 300
        B = 5
    if 'sea_con' in params_env['data_source']:
        A = 20
        B = 2
    if 'forescon' in params_env['data_source'] or 'sensorless' in params_env['data_source']:
        A = 0.1
        B = 1
    if  'shuttle' in params_env['data_source']:
        A = 200000
        B = 1

    if 'mniscon' in params_env['data_source']:
        A = 0.1
        B = 1
    if 'kdd' in params_env['data_source']:
        A = 0.3
        B = 1
    if 'water' in params_env['data_source']:
        A=10
        B=1

    set_seed(42)
    optimizer = torch.optim.Adam(list(ae_model.parameters()) + list(mlp.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()


    ###upsampling
    arr=X_train
    majority_class = arr[:1000]

    minority_classes = []
    start_index = 1000
    while start_index < len(X_train):
        end_index = start_index + 30
        minority_classes.append(X_train[start_index:end_index])
        start_index = end_index

    num_minorities = len(minority_classes)

    #smote
    class_counts = Counter(y_train)
    majority_class_label = class_counts.most_common(1)[0][0]  # Majority class label
    majority_class_size = class_counts[majority_class_label]
    # Initialize SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_train_up = X_resampled
    y_train_up = y_resampled


    # Check class distribution after oversampling
    print("Resampled dataset shape:", Counter(y_resampled))


# ######noupsampling: to do
    # Directly combine the majority class and the original minority classes without upsampling
    if 'noupsampling' in params_env['ind']:
        X_train_up = np.vstack([majority_class] + minority_classes)
        # Update labels
        y_train_up = np.array(
            [0] * majority_class_size + [i + 1 for i in range(num_minorities) for _ in range(len(minority_classes[i]))])


    X_train = torch.tensor(X_train_up, dtype=torch.float32)
    y_train = torch.tensor(y_train_up, dtype=torch.long)
    # Create Dataset and DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    ae_model.train()
    mlp.train()


    for epoch in range(num_epochs):
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()

            # Diffusion model forward pass
            recon_batch, z = ae_model(data)

            # Classifier forward pass
            if params_env['ae']=='noembed':
                output = mlp(data)
            else:
                output = mlp(z)

            # ** Calculate pseudo labels **
            with torch.no_grad():
                pseudo_labels = output.argmax(dim=1).detach()  # Use max probability class as pseudo label
                confidence = F.softmax(output, dim=1).max(dim=1)[0]  # Get highest confidence

            # Calculate loss
            if 'kdd' in params_env['data_source'] or 'vicon' in params_env['data_source'] or 'forescon' in params_env['data_source'] or 'blob' in params_env['data_source'] or 'water' in params_env['data_source'] or 'mniscon' in params_env['data_source'] or 'sea_con' in params_env['data_source']  in params_env['data_source'] or 'sensorless' in params_env['data_source']  or 'shuttle' in params_env['data_source']:
                recon_loss = F.mse_loss(recon_batch, data)


            clf_loss = criterion(output, target)
            recon_loss=recon_loss/A
            clf_loss=clf_loss/B

            # loss = 0.1 * recon_loss + 0.9 * clf_loss
            if params_env['com']== '28':
                loss = 0.2*recon_loss + 0.8*clf_loss
            elif params_env['com']== '55':
                loss = 0.5*recon_loss + 0.5*clf_loss
            elif params_env['com']== '82':
                loss = 0.8 * recon_loss+ 0.2 * clf_loss

            # Backward propagation and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        print(
            f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}, Accuracy: {100. * correct / total:.2f}%')


    return ae_model, mlp,


def lookahead_predictions_aemlp(t, d_env, ae_model, mlp):

    print('prediction',t,t + d_env['update_time'])
    lookahead_examples=d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]
    lookahead_examples = torch.tensor(lookahead_examples, dtype=torch.float32)
    lookahead_pred_class, logits_output, recon_loss, all_pro,_ = predict_with_ae_mlp(ae_model, mlp,lookahead_examples, d_env)
    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)


    return lookahead_pred_class,logits_output.cpu().detach().numpy(),recon_loss,all_pro

def lookahead_predictions_aemlp_noembed(t, d_env, ae_model, mlp):

    print('prediction',t,t + d_env['update_time'])
    lookahead_examples=d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]
    lookahead_examples = torch.tensor(lookahead_examples, dtype=torch.float32)
    lookahead_pred_class,logits_output,recon_loss=predict_with_ae_mlp_noembed(ae_model, mlp, lookahead_examples,d_env)
    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class,logits_output.cpu().detach().numpy(),recon_loss
#forest: noise_threshold=0.7, dbscan_eps=0.5, dbscan_min_samples=5,



def correct_and_resample_in_embedding_keep_nearest(
    ae_model,
    X_train_np,        # (N, D)
    y_train_np,        # (N,)
    keep_ratio=0.95,   # Keep only the "closest" top 95% of each class
    distance="mahalanobis",  # 'mahalanobis' | 'euclidean'
    standardize_Z=True,      # Whether to standardize embeddings for stable distance metrics
    do_smote=True,
    smote_kwargs=None,
    device="cpu",
    verbose=True,
):
    """
    Steps:
    1) Get embeddings Z using encoder;
    2) For each class in Z, compute distance of each sample to "class center" (Mahalanobis or Euclidean);
    3) Keep only the top keep_ratio percentage of samples with smallest distances (samples "closer to class center"), treat the rest as outliers and discard;
    4) Assemble X_core, y_core back in input space; (Optionally) perform SMOTE in input space;
    Return: X_resampled_np, y_resampled_np
    """
    # ---- 1) Encode to embedding space ----
    ae_model.eval()
    with torch.no_grad():
        Z_all = ae_model.encoder(torch.tensor(X_train_np, dtype=torch.float32, device=device)).cpu().numpy()

    # Optional: standardize Z (recommended to avoid distance distortion from large variance dimensions)
    if standardize_Z:
        scaler = StandardScaler()
        Z_all_std = scaler.fit_transform(Z_all)
    else:
        Z_all_std = Z_all

    classes = np.unique(y_train_np)
    counts = np.bincount(y_train_np)
    keep_indices = []

    if verbose:
        print("\n[Nearest-keep correction in embedding]")
        print(f"distance={distance}, keep_ratio={keep_ratio:.3f}, standardize_Z={standardize_Z}")
        print("Class distributions:", {int(c): int(counts[c]) for c in classes})

    for cls in classes:
        idx_cls = np.where(y_train_np == cls)[0]
        Z_cls = Z_all_std[idx_cls]

        # If too few samples, keep all (to prevent denominator too small)
        if len(Z_cls) <= max(5, int(1/(1-keep_ratio))):
            keep_indices.extend(idx_cls.tolist())
            if verbose:
                print(f"Class {int(cls)}: too few ({len(Z_cls)}), keep all.")
            continue

        # ---- 2) Compute distance to class center ----
        mu = Z_cls.mean(axis=0, keepdims=True)

        if distance.lower().startswith("mahal"):
            # Mahalanobis: use empirical covariance; fall back to Euclidean if non-invertible/ill-conditioned
            try:
                cov = EmpiricalCovariance().fit(Z_cls)
                diff = Z_cls - mu
                # mahalanobis distance squared = diagonal of diff * cov^{-1} * diff^T
                # directly use cov.mahalanobis returns Mahalanobis distance in Euclidean sense
                d = cov.mahalanobis(Z_cls)  # shape (n_cls,)
            except Exception:
                # Fallback: Euclidean
                diff = Z_cls - mu
                d = np.sqrt((diff**2).sum(axis=1))
        else:
            # Euclidean
            diff = Z_cls - mu
            d = np.sqrt((diff**2).sum(axis=1))

        # ---- 3) Keep only top keep_ratio percentage of "closest samples" ----
        k = max(1, int(np.floor(len(Z_cls) * keep_ratio)))
        # argsort distance, smallest to largest; take first k
        keep_local = np.argsort(d)[:k]
        chosen = idx_cls[keep_local]
        keep_indices.extend(chosen.tolist())

        if verbose:
            thr = np.sort(d)[k-1]
            print(f"Class {int(cls)}: total={len(Z_cls)} keep={k} "
                  f"({keep_ratio*100:.1f}%), dist_threshold={thr:.5f}")

    keep_indices = np.array(sorted(set(keep_indices)))
    X_core = X_train_np[keep_indices]
    y_core = y_train_np[keep_indices]

    if verbose:
        print("\n[After nearest-keep] Class distribution:")
        uq, cnts = np.unique(y_core, return_counts=True)
        for c, n in zip(uq, cnts):
            print(f"  Class {int(c)}: {int(n)} samples")

    # ---- 4) (Optional) Perform SMOTE in input space ----
    if do_smote and len(np.unique(y_core)) >= 2:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, **(smote_kwargs or {}))
        X_res, y_res = smote.fit_resample(X_core, y_core)
        if verbose:
            uq2, cnts2 = np.unique(y_res, return_counts=True)
            print("\n[After SMOTE] Class distribution:")
            for c, n in zip(uq2, cnts2):
                print(f"  Class {int(c)}: {int(n)} samples")
    else:
        X_res, y_res = X_core, y_core

    return X_res, y_res


def correct_and_resample_in_embedding(
    ae_model,
    X_train_np,        # numpy array, shape (N, D)
    y_train_np,        # numpy array, shape (N,)
    noise_threshold=0.9,
    dbscan_eps=0.5,
    dbscan_min_samples=5,
    do_smote=True,
    smote_kwargs=None,   # e.g., {'k_neighbors': 5}
    device="cpu"
):
    """
    Steps:
    1) Get Z using encoder;
    2) For each minority class in Z, perform DBSCAN, keep indices of "core points" (if noise ratio too high, keep all);
    3) Use these indices to get back to input space, obtain X_core, y_core;
    4) Directly perform SMOTE on X_core, y_core in input space (optional);
    Return: X_resampled_np, y_resampled_np
    """
    ae_model.eval()
    with torch.no_grad():
        Z_all = ae_model.encoder(torch.tensor(X_train_np, dtype=torch.float32, device=device)).cpu().numpy()

    classes = np.unique(y_train_np)
    counts = np.bincount(y_train_np)
    majority_cls = int(np.argmax(counts))

    # Majority class: keep all samples directly (in input space)
    keep_indices = list(np.where(y_train_np == majority_cls)[0])

    print("\n[DBSCAN correction results]")
    print(f"Majority class {majority_cls}: keep {len(keep_indices)}/{counts[majority_cls]} samples")

    for cls in classes:
        if cls == majority_cls or counts[cls] == 0:
            continue

        idx_cls = np.where(y_train_np == cls)[0]
        Z_min = Z_all[idx_cls]

        if len(Z_min) <= dbscan_min_samples:
            core_idx_local = np.arange(len(idx_cls))  # Too few, keep all
        else:
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(Z_min)
            core_mask = (db.labels_ != -1)
            noise_ratio = (db.labels_ == -1).mean()

            # Too much noise, avoid emptying class → keep all
            if noise_ratio > noise_threshold or core_mask.sum() == 0:
                core_idx_local = np.arange(len(idx_cls))
                print('noise ratio,',noise_ratio)
                print('sum of -1:', core_mask.sum() )
                print('too much noise, keep all')
            else:
                core_idx_local = np.where(core_mask)[0]

        # Print number of samples retained for this class
        print(f"Class {cls}: original {len(idx_cls)} → keep {len(core_idx_local)} samples")

        # Add "core points" global indices to keep list
        keep_indices.extend(idx_cls[core_idx_local].tolist())

    keep_indices = np.array(sorted(set(keep_indices)))
    X_core = X_train_np[keep_indices]
    y_core = y_train_np[keep_indices]

    # If fewer than two classes, skip SMOTE (to avoid errors)
    if do_smote and len(np.unique(y_core)) >= 2:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, **(smote_kwargs or {}))
        X_res, y_res = smote.fit_resample(X_core, y_core)
    else:
        X_res, y_res = X_core, y_core

    print("\n[After correction] Class distribution:")
    unique, counts = np.unique(y_core, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples")

    return X_res, y_res

def _class_density_via_knn_input(X_cls, k=10):
    """
    Use median k-th nearest neighbor distance to characterize sparsity, then take reciprocal as density proxy.
    Return density_proxy, med_kNN
    """
    n = len(X_cls)
    k_eff = min(k, max(2, n-1))   # Prevent out-of-bounds for small samples
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(X_cls)
    dists, _ = nbrs.kneighbors(X_cls)
    kth = dists[:, -1]                   # k-th nearest neighbor distance
    med_kNN = float(np.median(kth))
    density = 1.0 / max(med_kNN, 1e-8)   # Larger value means denser
    return density, med_kNN


def _class_density_via_knn(Z_cls, k=10):
    """
    Use median k-th nearest neighbor distance to characterize sparsity, then take reciprocal as density proxy.
    Return density_proxy, med_kNN
    """
    n = len(Z_cls)
    k_eff = min(k, max(2, n-1))   # Prevent out-of-bounds for small samples
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(Z_cls)
    dists, _ = nbrs.kneighbors(Z_cls)
    kth = dists[:, -1]                   # k-th nearest neighbor distance
    med_kNN = float(np.median(kth))
    density = 1.0 / max(med_kNN, 1e-8)   # Larger value means denser
    return density, med_kNN

def correct_and_resample_in_embedding_keep_nearest_minority_density_adaptive(
        t,
    ae_model,
    X_train_np,        # (N, D)
    y_train_np,        # (N,)
    keep_min=0.85,     # Minimum retention ratio (lowest density class)
    keep_max=0.99,     # Maximum retention ratio (highest density class)
    knn_k=10,          # k for density calculation
    distance="mahalanobis",  # 'mahalanobis' | 'euclidean'
    standardize_Z=True,
    do_smote=True,
    smote_kwargs=None,
    device="cpu",
    verbose=True,
    min_keep_per_class=5,     # Minimum number of samples to keep per minority class
    return_indices=False,     # If True, additionally return kept/dropped global indices per class
):
    """
    Only filter "minority classes"; retention ratio for each minority class is adaptively determined by "class density".
    High density → keep_ratio closer to keep_max; low density → keep_ratio closer to keep_min.
    Distance used to determine ordering of "nearest samples" (Mahalanobis/Euclidean).

    Additionally: if 3000 <= t <= 3500, save removed samples (features+labels+distance to center) as CSV: removed_samples_t{t}.csv

    Return:
        X_res, y_res, (optional) keep_drop_map  # keep_drop_map: {cls: {"kept":[...], "removed":[...]}}
    """
    assert 0.0 < keep_min <= keep_max <= 1.0

    # ---- 1) Encode to embedding space ----
    ae_model.eval()
    with torch.no_grad():
        Z_all = ae_model.encoder(torch.tensor(X_train_np, dtype=torch.float32, device=device)).cpu().numpy()

    # Optional standardization, stabilize distance/knn
    Z_all_std = StandardScaler().fit_transform(Z_all) if standardize_Z else Z_all

    classes = np.unique(y_train_np)
    binc = np.bincount(y_train_np.astype(int))
    majority_cls = int(np.argmax(binc))

    # ---- 2) Estimate density and map to keep_ratio ----
    class_density = {}
    for cls in classes:
        idx = np.where(y_train_np == cls)[0]
        Zc = Z_all_std[idx]
        if len(Zc) <= 2:
            class_density[int(cls)] = 0.0
        else:
            dens, _ = _class_density_via_knn(Zc, k=knn_k)
            class_density[int(cls)] = dens

    dens_vals = np.array(list(class_density.values()), dtype=float)
    dmin, dmax = dens_vals.min(), dens_vals.max()
    if dmax > dmin:
        class_density_norm = {c: (class_density[c]-dmin)/(dmax-dmin) for c in class_density}
    else:
        class_density_norm = {c: 0.5 for c in class_density}

    class_keep_ratio = {
        c: (keep_min + class_density_norm[c] * (keep_max - keep_min))
        for c in class_density_norm
    }

    if verbose:
        print("\n[Density-adaptive keep ratios]")
        for c in sorted(class_keep_ratio):
            print(f"Class {c}: density={class_density[c]:.5f}, "
                  f"norm={class_density_norm[c]:.3f}, keep={class_keep_ratio[c]:.3f}")
        print(f"Majority class (no filtering): {majority_cls}")

    # ---- 3) Class-wise filtering + record removed samples ----
    keep_indices = []
    keep_drop_map = {}      # {cls: {"kept":[...], "removed":[...]}}
    removed_samples = []    # Store sample content (no global index dependency)

    for cls in classes:
        idx_cls = np.where(y_train_np == cls)[0]   # Global indices within class
        Z_cls = Z_all_std[idx_cls]

        # Majority class: no filtering
        if int(cls) == majority_cls:
            keep_indices.extend(idx_cls.tolist())
            keep_drop_map[int(cls)] = {"kept": idx_cls.tolist(), "removed": []}
            if verbose:
                print(f"Class {int(cls)}: majority, keep all ({len(idx_cls)})")
            continue

        # Minority class too few: keep all
        if len(Z_cls) <= max(min_keep_per_class, int(1/(1-keep_min))):
            keep_indices.extend(idx_cls.tolist())
            keep_drop_map[int(cls)] = {"kept": idx_cls.tolist(), "removed": []}
            if verbose:
                print(f"Class {int(cls)}: too few ({len(Z_cls)}), keep all.")
            continue

        # Distance to class center
        mu = Z_cls.mean(axis=0, keepdims=True)
        if distance.lower().startswith("mahal"):
            try:
                cov = EmpiricalCovariance().fit(Z_cls)
                d = cov.mahalanobis(Z_cls)
            except Exception:
                diff = Z_cls - mu
                d = np.sqrt((diff**2).sum(axis=1))
        else:
            diff = Z_cls - mu
            d = np.sqrt((diff**2).sum(axis=1))

        keep_ratio_c = float(class_keep_ratio[int(cls)])
        k_keep = max(min_keep_per_class, int(np.floor(len(Z_cls) * keep_ratio_c)))
        keep_local = np.argsort(d)[:k_keep]
        drop_local = np.setdiff1d(np.arange(len(idx_cls)), keep_local)

        chosen = idx_cls[keep_local]
        removed = idx_cls[drop_local]

        keep_indices.extend(chosen.tolist())
        keep_drop_map[int(cls)] = {"kept": chosen.tolist(), "removed": removed.tolist()}

        # —— Save content of removed samples (distance & features) ——
        for loc, r in zip(drop_local, removed):
            removed_samples.append(
                dict(
                    cls=int(cls),
                    distance=float(d[loc]),
                    **{f"f{i}": X_train_np[r, i] for i in range(X_train_np.shape[1])}
                )
            )

        if verbose:
            thr = np.sort(d)[k_keep-1]
            print(f"\n[Class {int(cls)}] total={len(Z_cls)} keep={k_keep} "
                  f"({keep_ratio_c*100:.1f}%)  dist_threshold={thr:.5f}  removed={len(removed)}")

    # ---- 4) Conditionally save removed samples CSV ----
    if 3000 <= t <= 3500 and len(removed_samples) > 0:
        out_df = pd.DataFrame(removed_samples)
        out_path = f"removed_samples_t{t}.csv"
        out_df.to_csv(out_path, index=False)
        if verbose:
            print(f"[Saved removed samples] -> {out_path}")

    # ---- 5) Merge kept + optional SMOTE ----
    keep_indices = np.array(sorted(set(keep_indices)))
    X_core = X_train_np[keep_indices]
    y_core = y_train_np[keep_indices]

    if verbose:
        print("\n[After density-adaptive filtering] Class distribution:")
        uq, cnts = np.unique(y_core, return_counts=True)
        for c, n in zip(uq, cnts):
            print(f"  Class {int(c)}: {int(n)} samples")

    if do_smote and len(np.unique(y_core)) >= 2:
        smote = SMOTE(random_state=42, **(smote_kwargs or {})) if smote_kwargs else SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_core, y_core)
        if verbose:
            uq2, cnts2 = np.unique(y_res, return_counts=True)
            print("\n[After SMOTE] Class distribution:")
            for c, n in zip(uq2, cnts2):
                print(f"  Class {int(c)}: {int(n)} samples")
    else:
        X_res, y_res = X_core, y_core

    if return_indices:
        return X_res, y_res, keep_drop_map
    else:
        return X_res, y_res
def correct_and_resample_in_input_keep_nearest_minority_density_adaptive(
    X_train_np,        # (N, D)
    y_train_np,        # (N,)
    keep_min=0.85,     # Retention ratio for lowest density class
    keep_max=0.99,     # Retention ratio for highest density class
    knn_k=10,          # k for density estimation
    distance="mahalanobis",  # 'mahalanobis' | 'euclidean'
    standardize_X=True,      # Whether to standardize input space (strongly recommended)
    do_smote=True,
    smote_kwargs=None,
    verbose=True,
    min_keep_per_class=5,    # Minimum to keep per minority class
):
    """
    All in input space:
    1) Estimate density per class in input space (reciprocal of median k-nearest neighbor distance) → map to retention ratio [keep_min, keep_max];
    2) For minority classes only, keep the closest keep_ratio_c by distance to class center;
    3) (Optional) Perform SMOTE in input space.
    """
    assert 0.0 < keep_min <= keep_max <= 1.0

    # 1) Standardize input (recommended to avoid feature scale inconsistency)
    X_all_std = StandardScaler().fit_transform(X_train_np) if standardize_X else X_train_np

    classes = np.unique(y_train_np)
    binc = np.bincount(y_train_np.astype(int))
    majority_cls = int(np.argmax(binc))

    # 2) Estimate per-class density and map to retention ratio
    class_density = {}
    for cls in classes:
        idx = np.where(y_train_np == cls)[0]
        Xc = X_all_std[idx]
        dens, _ = _class_density_via_knn_input(Xc, k=knn_k)
        class_density[int(cls)] = dens

    dens_vals = np.array(list(class_density.values()), dtype=float)
    dmin, dmax = dens_vals.min(), dens_vals.max()
    if dmax > dmin:
        class_density_norm = {c: (class_density[c]-dmin)/(dmax-dmin) for c in class_density}
    else:
        class_density_norm = {c: 0.5 for c in class_density}

    class_keep_ratio = {
        c: (keep_min + class_density_norm[c]*(keep_max - keep_min))
        for c in class_density_norm
    }

    if verbose:
        print("\n[Input-space density-adaptive keep ratios]")
        for c in sorted(class_keep_ratio):
            print(f"Class {c}: density={class_density[c]:.5f}, "
                  f"norm={class_density_norm[c]:.3f}, keep={class_keep_ratio[c]:.3f}")
        print(f"Majority class (no filtering): {majority_cls}")

    # 3) Filter only minority classes: keep closest by distance to class center
    keep_indices = []
    for cls in classes:
        idx_cls = np.where(y_train_np == cls)[0]
        X_cls = X_all_std[idx_cls]

        # Majority class not filtered
        if int(cls) == majority_cls:
            keep_indices.extend(idx_cls.tolist())
            if verbose:
                print(f"Class {int(cls)}: majority, keep all ({len(idx_cls)})")
            continue

        # Too few samples → keep all
        if len(X_cls) <= max(min_keep_per_class, int(1/(1 - keep_min))):
            keep_indices.extend(idx_cls.tolist())
            if verbose:
                print(f"Class {int(cls)}: too few ({len(X_cls)}), keep all.")
            continue

        # Distance to class center
        mu = X_cls.mean(axis=0, keepdims=True)
        if distance.lower().startswith("mahal"):
            try:
                cov = EmpiricalCovariance().fit(X_cls)
                d = cov.mahalanobis(X_cls)
            except Exception:
                diff = X_cls - mu
                d = np.sqrt((diff**2).sum(axis=1))
        else:
            diff = X_cls - mu
            d = np.sqrt((diff**2).sum(axis=1))

        keep_ratio_c = float(class_keep_ratio[int(cls)])
        k_keep = max(min_keep_per_class, int(np.floor(len(X_cls) * keep_ratio_c)))
        keep_local = np.argsort(d)[:k_keep]
        chosen = idx_cls[keep_local]
        keep_indices.extend(chosen.tolist())

        if verbose:
            thr = np.sort(d)[k_keep - 1]
            print(f"Class {int(cls)}: total={len(X_cls)} keep={k_keep} "
                  f"({keep_ratio_c*100:.1f}%), dist_threshold={thr:.5f}")

    keep_indices = np.array(sorted(set(keep_indices)))
    X_core = X_train_np[keep_indices]   # Note: back to original scale data
    y_core = y_train_np[keep_indices]

    if verbose:
        print("\n[After input-space filtering] Class distribution:")
        uq, cnts = np.unique(y_core, return_counts=True)
        for c, n in zip(uq, cnts):
            print(f"  Class {int(c)}: {int(n)} samples")

    # 4) (Optional) Perform SMOTE in input space
    if do_smote and len(np.unique(y_core)) >= 2:
        from imblearn.over_sampling import SMOTE
        # Here directly use previously discussed robust wrapper, ensure oversampling to majority class size and auto-adjust k_neighbors
        from collections import Counter
        cls_cnt = Counter(y_core)
        maj_n = max(cls_cnt.values())
        sampling_strategy = {c: maj_n for c, n in cls_cnt.items() if n < maj_n}
        if sampling_strategy:
            min_minority = min(cls_cnt[c] for c in sampling_strategy)
            base_k = (smote_kwargs or {}).get('k_neighbors', 5)
            k_neighbors = max(1, min(base_k, min_minority - 1))
            smote = SMOTE(sampling_strategy=sampling_strategy,
                          k_neighbors=k_neighbors,
                          random_state=(smote_kwargs or {}).get('random_state', 42))
            X_res, y_res = smote.fit_resample(X_core, y_core)
        else:
            X_res, y_res = X_core, y_core
        if verbose:
            uq2, cnts2 = np.unique(y_res, return_counts=True)
            print("\n[After SMOTE] Class distribution:")
            for c, n in zip(uq2, cnts2):
                print(f"  Class {int(c)}: {int(n)} samples")
    else:
        X_res, y_res = X_core, y_core

    return X_res, y_res



#############
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance, MinCovDet, LedoitWolf, OAS

# ===================== Utility functions =====================

def geometric_median(X, eps=1e-6, max_iter=256):
    """
    Geometric median (Weiszfeld algorithm), more robust to outliers than mean.
    X: (n, d)
    """
    y = X.mean(axis=0)
    for _ in range(max_iter):
        diff = X - y
        dist = np.linalg.norm(diff, axis=1) + 1e-12
        inv = 1.0 / np.maximum(dist, eps)
        T = (inv[:, None] * X).sum(axis=0) / inv.sum()
        if np.linalg.norm(y - T) < eps:
            return T
        y = T
    return y

def robust_center_and_cov(
    X,
    center_method="geomed",     # 'geomed' | 'median' | 'mean'
    cov_method="mcd",           # 'mcd' | 'oas' | 'lw' | 'empirical'
    iter_trim_q=None,           # e.g., 0.8 means keep top 80% closest points then re-estimate
    iter_times=1,
    random_state=42
):
    """
    Robust center + covariance estimation. Returns (mu, cov_estimator).
    cov_estimator must support mahalanobis() or have precision_.
    """
    X_curr = X.copy()
    for it in range(max(1, iter_times)):
        # 1) Center
        if center_method == "geomed":
            mu = geometric_median(X_curr)
        elif center_method == "median":
            mu = np.median(X_curr, axis=0)
        else:
            mu = X_curr.mean(axis=0)

        # 2) Covariance
        if cov_method == "mcd":
            cov = MinCovDet(random_state=random_state, assume_centered=False).fit(X_curr)
        elif cov_method == "oas":
            cov = OAS(assume_centered=False).fit(X_curr)
        elif cov_method == "lw":
            cov = LedoitWolf(assume_centered=False).fit(X_curr)
        else:
            cov = EmpiricalCovariance().fit(X_curr)

        # 3) Optional tail trimming → re-estimate next iteration
        if iter_trim_q is not None and it < iter_times:
            try:
                d = cov.mahalanobis(X_curr)
            except Exception:
                d = np.linalg.norm(X_curr - mu, axis=1)
            q = float(iter_trim_q)
            q = min(max(q, 0.5), 0.95)
            thr = np.quantile(d, q)
            X_curr = X_curr[d <= thr]
            if X_curr.shape[0] < max(10, X.shape[1] + 2):
                break
    return mu, cov

def distances_to_class_center_robust(
    X_cls,
    center_method="geomed",
    cov_method="mcd",
    iter_trim_q=0.8,
    iter_times=1,
    use_mahalanobis=True,
    random_state=42
):
    """
    For single class samples, compute distance to robust center (prefer Mahalanobis, fallback Euclidean).
    Return d, mu_hat, cov_hat
    """
    mu, cov = robust_center_and_cov(
        X_cls,
        center_method=center_method,
        cov_method=cov_method,
        iter_trim_q=iter_trim_q,
        iter_times=iter_times,
        random_state=random_state
    )
    if use_mahalanobis:
        try:
            d = cov.mahalanobis(X_cls)
        except Exception:
            d = np.linalg.norm(X_cls - mu, axis=1)
    else:
        d = np.linalg.norm(X_cls - mu, axis=1)
    return d, mu, cov

def class_density_via_knn_median(Xc, k=10, eps=1e-12):
    """
    Use reciprocal of within-class kNN median distance to approximate class density (return class average density).
    Note: Use Euclidean distance for density estimation (robust, simple).
    """
    n = Xc.shape[0]
    if n <= 1:
        return 0.0
    k = min(max(1, k), max(1, n-1))
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(Xc)
    dist, idx = nbrs.kneighbors(Xc)
    # Column 0 is self-distance (0), discard
    dist = dist[:, 1:]
    # Each sample: kNN median distance
    med = np.median(dist, axis=1)
    dens_each = 1.0 / (med + eps)
    return float(np.mean(dens_each))

# ===================== Main function =====================


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler


#
def correct_and_resample_in_input_keep_nearest_minority_density_adaptive_new(
    X_train_np,        # (N, D)
    y_train_np,        # (N,)
    keep_min=0.85,     # Lower bound retention ratio for lowest density class
    keep_max=0.99,     # Upper bound retention ratio for highest density class
    knn_k=10,          # k for density estimation
    distance="mahalanobis",  # 'mahalanobis' | 'euclidean' (for center filtering)
    standardize_X=True,      # Whether to standardize input space (strongly recommended)
    do_smote=True,           # Whether to perform SMOTE oversampling
    smote_kwargs=None,
    verbose=True,
    min_keep_per_class=5,    # Minimum to keep per minority class
    # ---- Sparsity compensation & mixing weights ----
    use_relative_density=True,   # Use relative density compensation (rho * s_c)
    mix_with_class_size=True,    # Mix weight with class size
    lambda_density=0.7,          # Mixing weight: density proportion
    # ---- Robust center/covariance parameters ----
    center_method="geomed",      # 'geomed' | 'median' | 'mean'
    cov_method="mcd",            # 'mcd' | 'oas' | 'lw' | 'empirical'
    iter_trim_q=0.8,             # Trimming quantile (None to disable)
    iter_times=1,                # Trim-reestimate iterations
    random_state=42,
    # ---- New: removed samples saving related ----
    save_removed=True,                        # Whether to save removed samples
    removed_save_path="removed_samples.csv"   # Save path (CSV)
):
    """
    Input space data correction and resampling pipeline (robust version):
      1) Standardization
      2) Estimate density per class (reciprocal of kNN median distance) + (optional) relative density compensation
      3) Map density to retention ratio, and mix with class size weighted
      4) Filter only minority classes: keep closest r_c proportion by robust center distance (Mahalanobis preferred)
      5) (Optional) SMOTE oversampling to majority class size
    Additional functionality:
      - Save all removed samples to CSV file
    Return: (X_res, y_res)

    Dependencies (need to be provided in project):
      - class_density_via_knn_median(Xc, k)
      - geometric_median(Xc)
      - distances_to_class_center_robust(..., use_mahalanobis, ...)
    """
    assert 0.0 < keep_min <= keep_max <= 1.0
    smote_kwargs = smote_kwargs or {}

    # 1) Standardization
    if standardize_X:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_train_np)
    else:
        X_std = X_train_np

    y = y_train_np.astype(int)
    classes = np.unique(y)
    binc = np.bincount(y)
    majority_cls = int(np.argmax(binc))

    # 2) Class density + sparsity compensation
    class_density = {}
    class_scale = {}  # Within-class scale s_c
    for cls in classes:
        idx = np.where(y == cls)[0]
        Xc = X_std[idx]
        rho_c = class_density_via_knn_median(Xc, k=knn_k)
        class_density[int(cls)] = rho_c
        mu_c = geometric_median(Xc) if Xc.shape[0] > 0 else np.zeros(Xc.shape[1])
        s_c = float(np.mean(np.linalg.norm(Xc - mu_c, axis=1))) if Xc.shape[0] > 0 else 0.0
        class_scale[int(cls)] = s_c

    if use_relative_density:
        class_density_comp = {c: class_density[c] * max(class_scale[c], 1e-12) for c in class_density}
    else:
        class_density_comp = dict(class_density)

    dens_vals = np.array(list(class_density_comp.values()), dtype=float)
    dmin, dmax = dens_vals.min(), dens_vals.max()
    if dmax > dmin:
        class_density_norm = {c: (class_density_comp[c] - dmin) / (dmax - dmin) for c in class_density_comp}
    else:
        class_density_norm = {c: 0.5 for c in class_density_comp}

    size_norm = {c: (np.sum(y == c) / np.max([np.sum(y == cc) for cc in classes])) for c in classes}

    class_keep_ratio = {}
    for c in classes:
        dens_part = class_density_norm[int(c)]
        if mix_with_class_size:
            mixed = lambda_density * dens_part + (1.0 - lambda_density) * size_norm[int(c)]
        else:
            mixed = dens_part
        r_c = keep_min + mixed * (keep_max - keep_min)
        class_keep_ratio[int(c)] = float(np.clip(r_c, keep_min, keep_max))

    if verbose:
        print("\n[Class density and retention ratios (with compensation/mixing)]")
        for c in sorted(classes):
            print(f"Class {int(c)}: rho={class_density[int(c)]:.6f}, "
                  f"s_c={class_scale[int(c)]:.6f}, "
                  f"rho'=({('on' if use_relative_density else 'off')}): "
                  f"{class_density_comp[int(c)]:.6f}, keep={class_keep_ratio[int(c)]:.3f}")
        print(f"Majority class (no filtering): {majority_cls}")

    # 4) Filter only minority classes
    keep_indices = []
    use_mahal = (distance.lower().startswith("mahal"))
    for cls in classes:
        idx_cls = np.where(y == cls)[0]
        X_cls = X_std[idx_cls]

        if int(cls) == majority_cls:
            keep_indices.extend(idx_cls.tolist())
            if verbose:
                print(f"Class {int(cls)}: majority, keep all ({len(idx_cls)})")
            continue

        if len(X_cls) <= max(min_keep_per_class, int(1/(1 - keep_min))):
            keep_indices.extend(idx_cls.tolist())
            if verbose:
                print(f"Class {int(cls)}: too few ({len(X_cls)}), keep all.")
            continue

        d, mu_hat, cov_hat = distances_to_class_center_robust(
            X_cls,
            center_method=center_method,
            cov_method=cov_method,
            iter_trim_q=iter_trim_q,
            iter_times=iter_times,
            use_mahalanobis=use_mahal,
            random_state=random_state
        )

        keep_ratio_c = float(class_keep_ratio[int(cls)])
        k_keep = max(min_keep_per_class, int(np.floor(len(X_cls) * keep_ratio_c)))
        order = np.argsort(d)
        keep_local = order[:k_keep]
        chosen = idx_cls[keep_local]
        keep_indices.extend(chosen.tolist())

        if verbose:
            thr = np.sort(d)[k_keep - 1]
            print(f"Class {int(cls)}: total={len(X_cls)} keep={k_keep} "
                  f"({keep_ratio_c*100:.1f}%), dist_threshold={thr:.6f}")

    keep_indices = np.array(sorted(set(keep_indices)))
    X_core = X_train_np[keep_indices]
    y_core = y[keep_indices]

    # ---- New: Save all removed samples ----
    if save_removed:
        all_idx = np.arange(len(y))
        removed_indices = np.setdiff1d(all_idx, keep_indices, assume_unique=True)

        if removed_indices.size > 0:
            try:
                df_removed = pd.DataFrame(
                    X_train_np[removed_indices],
                    index=removed_indices
                )
                df_removed["label"] = y[removed_indices]
                df_removed.index.name = "index"
                df_removed.to_csv(removed_save_path)
                if verbose:
                    print(f"[INFO] Saved {len(removed_indices)} removed samples "
                          f"to: {removed_save_path}")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to save removed samples: {e}")
        else:
            if verbose:
                print(f"[INFO] No samples were removed.")

    if verbose:
        print("\n[After input-space filtering] Class distribution:")
        uq, cnts = np.unique(y_core, return_counts=True)
        for c, n in zip(uq, cnts):
            print(f"  Class {int(c)}: {int(n)}")

    # 5) Optional SMOTE
    if do_smote and len(np.unique(y_core)) >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            cls_cnt = Counter(y_core)
            maj_n = max(cls_cnt.values())
            sampling_strategy = {c: maj_n for c, n in cls_cnt.items() if n < maj_n}
            if sampling_strategy:
                min_minority = min(cls_cnt[c] for c in sampling_strategy)
                base_k = smote_kwargs.get('k_neighbors', 5)
                k_neighbors = max(1, min(base_k, min_minority - 1))
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=smote_kwargs.get('random_state', random_state)
                )
                X_res, y_res = smote.fit_resample(X_core, y_core)
            else:
                X_res, y_res = X_core, y_core
        except Exception as e:
            if verbose:
                print(f"[WARN] SMOTE failed, skipping oversampling: {e}")
            X_res, y_res = X_core, y_core

        if verbose:
            uq2, cnts2 = np.unique(y_res, return_counts=True)
            print("\n[After SMOTE] Class distribution:")
            for c, n in zip(uq2, cnts2):
                print(f"  Class {int(c)}: {int(n)}")
    else:
        X_res, y_res = X_core, y_core

    return X_res, y_res


#########
def correct_and_resample_X_via_embedding_new(
    X_train_np,
    y_train_np,
    ae_model,
    device="cpu",
    keep_min=0.85,
    keep_max=0.99,
    knn_k=10,
    distance="mahalanobis",
    standardize_Z=True,
    do_smote=True,
    smote_kwargs=None,
    verbose=True,
    min_keep_per_class=5,
    use_relative_density=True,
    mix_with_class_size=True,
    lambda_density=0.7,
    center_method="geomed",
    cov_method="oas",
    iter_trim_q=0.8,
    iter_times=1,
    random_state=42,
    reduce_dim=None,
    decode_after_smote=True   # Whether to use decoder to map SMOTE-generated embeddings back to X
):
    """
    Perform filtering logic in embedding space, but ultimately return real input X subset/resampling results.
    """
    from collections import Counter
    import torch
    from sklearn.preprocessing import StandardScaler

    smote_kwargs = smote_kwargs or {}

    # 1) Encode to embedding
    ae_model.eval()
    with torch.no_grad():
        Z_all = ae_model.encoder(
            torch.tensor(X_train_np, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # 2) Optional dimensionality reduction
    if reduce_dim is not None and reduce_dim < Z_all.shape[1]:
        from sklearn.decomposition import PCA
        Z_all = PCA(n_components=reduce_dim, random_state=random_state).fit_transform(Z_all)

    # 3) Standardization
    Z_std = StandardScaler().fit_transform(Z_all) if standardize_Z else Z_all

    y = y_train_np.astype(int)
    classes = np.unique(y)
    binc = np.bincount(y)
    majority_cls = int(np.argmax(binc))

    # 4) Class density + compensation
    class_density, class_scale = {}, {}
    for cls in classes:
        idx = np.where(y == cls)[0]
        Zc = Z_std[idx]
        rho_c = class_density_via_knn_median(Zc, k=knn_k)
        class_density[int(cls)] = rho_c
        mu_c = geometric_median(Zc) if Zc.shape[0] > 0 else np.zeros(Zc.shape[1])
        s_c = float(np.mean(np.linalg.norm(Zc - mu_c, axis=1))) if Zc.shape[0] > 0 else 0.0
        class_scale[int(cls)] = s_c

    if use_relative_density:
        class_density_comp = {c: class_density[c] * max(class_scale[c], 1e-12) for c in class_density}
    else:
        class_density_comp = dict(class_density)

    dens_vals = np.array(list(class_density_comp.values()), dtype=float)
    dmin, dmax = dens_vals.min(), dens_vals.max()
    if dmax > dmin:
        class_density_norm = {c: (class_density_comp[c] - dmin) / (dmax - dmin) for c in class_density_comp}
    else:
        class_density_norm = {c: 0.5 for c in class_density_comp}

    size_norm = {c: (np.sum(y == c) / np.max([np.sum(y == cc) for cc in classes])) for c in classes}
    class_keep_ratio = {}
    for c in classes:
        dens_part = class_density_norm[int(c)]
        mixed = lambda_density * dens_part + (1.0 - lambda_density) * size_norm[int(c)] if mix_with_class_size else dens_part
        r_c = keep_min + mixed * (keep_max - keep_min)
        class_keep_ratio[int(c)] = float(np.clip(r_c, keep_min, keep_max))

    # 5) Filtering (compute distance in embedding space, but apply indices to X)
    keep_indices = []
    use_mahal = distance.lower().startswith("mahal")
    for cls in classes:
        idx_cls = np.where(y == cls)[0]
        Z_cls = Z_std[idx_cls]

        if int(cls) == majority_cls or len(Z_cls) <= max(min_keep_per_class, int(1/(1 - keep_min))):
            keep_indices.extend(idx_cls.tolist())
            continue

        d, mu_hat, cov_hat = distances_to_class_center_robust(
            Z_cls,
            center_method=center_method,
            cov_method=cov_method,
            iter_trim_q=iter_trim_q,
            iter_times=iter_times,
            use_mahalanobis=use_mahal,
            random_state=random_state
        )

        keep_ratio_c = float(class_keep_ratio[int(cls)])
        k_keep = max(min_keep_per_class, int(np.floor(len(Z_cls) * keep_ratio_c)))
        order = np.argsort(d)
        keep_local = order[:k_keep]
        chosen = idx_cls[keep_local]
        keep_indices.extend(chosen.tolist())

    keep_indices = np.array(sorted(set(keep_indices)))
    X_core = X_train_np[keep_indices]   # Take subset in X space
    y_core = y[keep_indices]
    Z_core = Z_all[keep_indices]

    # 6) Optional SMOTE
    if do_smote and len(np.unique(y_core)) >= 2:
        from imblearn.over_sampling import SMOTE
        cls_cnt = Counter(y_core)
        maj_n = max(cls_cnt.values())
        sampling_strategy = {c: maj_n for c, n in cls_cnt.items() if n < maj_n}
        if sampling_strategy:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=max(1, min(smote_kwargs.get('k_neighbors', 5), min(cls_cnt.values()) - 1)),
                random_state=smote_kwargs.get('random_state', random_state)
            )
            Z_res, y_res = smote.fit_resample(Z_core, y_core)

            if decode_after_smote:
                # Use decoder to map synthesized embeddings back to X space
                with torch.no_grad():
                    X_res = ae_model.decoder(
                        torch.tensor(Z_res, dtype=torch.float32, device=device)
                    ).cpu().numpy()
            else:
                # Directly return X_core subset, excluding synthesized samples
                X_res = X_core
                y_res = y_core
        else:
            X_res, y_res = X_core, y_core
    else:
        X_res, y_res = X_core, y_core

    return X_res, y_res



###########################################################################################
#                                           Run                                           #
###########################################################################################


def run_con(params_env):
    eps=10
    min_samples=5

    ######################
    # Init preq. metrics #
    ######################
    params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_arr(params_env)
    params_env['update_time'] = int(params_env['unsupervised_win_size'] * params_env['unsupervised_win_size_update'])
    params_env['preq_fading_factor'] = 0.99
    params_env['time_steps'] = params_env['data_arr'].shape[0]
    params_env['num_features'] = params_env['data_arr'].shape[1] - 1
    # params_env['num_classes'] = len(np.unique(params_env['data_arr'][:, -1]))
    params_env['num_classes'] = 4
    if 'blob' in params_env['data_source']:
        params_env['num_classes'] = 7
    if 'mniscon' in params_env['data_source']:
        params_env['num_classes'] = 10
    if 'water' in params_env['data_source']:
        params_env['num_classes'] = 3

    ######################
    # Init preq. metrics #
    ######################

    # general accuracy
    preq_general_accs = []
    preq_general_acc_n = 0.0
    preq_general_acc_s = 0.0

    # class accuracies
    keys = range(params_env['num_classes'])
    preq_class_accs = {k: [] for k in keys}
    preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))
    # gmean
    preq_gmeans = []

    ##pred
    class_result=[]
    pro_result=[]

    ###initialization for recall and specificity
    preq_recalls = np.zeros(params_env['time_steps'])
    preq_specificities = np.zeros(params_env['time_steps'])
    #
    preq_recall, preq_specificity = (1.0,) * 2  # NOTE: init to 1.0 not 0.0
    preq_recall_s, preq_recall_n = (0.0,) * 2
    preq_specificity_s, preq_specificity_n = (0.0,) * 2



 #################################################
    # Init unsupervised learning (VAE) #
    #################################################
    ae = None
    lookahead_pred_class = None

    # Initialize list to store anomalous class timesteps
    t_as_list = [[] for _ in range(10)]
    retrain_time = []
    class_an=1
    if 'water' in params_env['data_source']:
        class_an=0

    t_ap =0
    new_first=0


    mov_win_np_mul = deque(maxlen=30)
    ref_win_np_mul = deque(maxlen=30)
    euclidean_thresholds=[]
    q_unlabelled = deque( maxlen=1000)

    # Initialize other required structures
    ref_lof_aps=[]
    mov_win_aps = []
    ref_win_aps=[]
    normal_win = []
    recon_mov = []
    recon_ref = []
    threshold_mulclass=[]
    mov_np_retrain=deque(maxlen=1000)

    # Initialize list to store ref windows
    for i in range(class_an):
        ref_win_aps.append(deque(maxlen=30))
        mov_win_aps.append(deque(maxlen=30))
        ref_lof_aps.append(deque(maxlen=30))

    params_env['seed'] = 42
    params_env['random_state'] = np.random.RandomState(seed=params_env['seed'])


    if 'vicon' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_vib_ref()
    if 'blob' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_blob_ref()
    if 'sea_con' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_sea_ref()
    if 'forescon' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_forest_ref()
    if 'sensorless' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_sensor_ref()
    if 'shuttle' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_shuttle_ref()

    if 'mniscon' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_mnist_ref()
    if 'kdd' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_kdd_ref()
    if 'water' in params_env['data_source']:
        ref_win_np, ref_win_ap = load_water_ref(params_env)

    points = ref_win_np[:1000]
    max_distance = 0
    for point1, point2 in combinations(points, 2):
        distance = np.linalg.norm(point1 - point2)
        if distance > max_distance:
            max_distance = distance
    euclidean_thresholds.append(max_distance)

    if len(ref_win_ap)>0:
        ref_win_aps[0] = ref_win_ap[:30, :]
        ref_lof_aps[0] = ref_win_ap[:30, :]
        points = ref_lof_aps[0]
        max_distance = 0
        for point1, point2 in combinations(points, 2):
            distance = np.linalg.norm(point1 - point2)
            if distance > max_distance:
                max_distance = distance
        euclidean_thresholds.append(max_distance)




    print(euclidean_thresholds,'euclidean_thresholds')




###evaluation

    if params_env['method'] == 'clf':
        print('run with clf')
        ae_model, mlp, logits_output, ref_logit0, ref_logit1, threshold_mulclass0, threshold_mulclass1, recon_ref=pretrain_aemlp_mulincr(params_env)
        threshold_mulclass.append(threshold_mulclass0)
        if threshold_mulclass1 is not None:
            threshold_mulclass.append(threshold_mulclass1)
        if params_env['ae']=='noembed':
            lookahead_pred_class, logits_output, recon_loss = lookahead_predictions_aemlp_noembed(0, params_env,ae_model, mlp)
        else:
            lookahead_pred_class,logits_output,recon_loss,all_pro= lookahead_predictions_aemlp(0, params_env, ae_model, mlp)
        print('pretrain threshold', threshold_mulclass)

    if params_env['method'] == 'iforest':
        iforest = create_iforest(params_env)
        train_iforest(ref_win_np[:1000], iforest)
        lookahead_pred_class= lookahead_predictions_iforest(0, params_env, iforest)
    if params_env['method'] == 'lof':
        lof = create_lof(params_env)
        train_lof(ref_win_np[:1000], lof)
        lookahead_pred_class = lookahead_predictions_lof(0, params_env, lof)


####online
    for t in range(0, params_env['time_steps']):
        if t % 500 == 0:
            print('Time step: ', t)

        #################
        # Concept drift #
        #################

        # reset preq. metrics
        if 'drift' in params_env['data_source']:
            flag_reset_metric = False


            if isinstance(params_env['t_drift'], int) and t == params_env['t_drift']:
                flag_reset_metric = True
            elif isinstance(params_env['t_drift'], tuple) and \
                    (t == params_env['t_drift'][0] or t == params_env['t_drift'][1]):
                flag_reset_metric = True



            if flag_reset_metric:
                preq_general_acc_n = 0.0
                preq_general_acc_s = 0.0

                preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
                preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
                preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

         ###############
        # Get example #
        ###############


        xy = params_env['data_arr'][t, :]

        x = xy[:-1]  # of shape (n,)
        x = np.reshape(x, (1, x.shape[0]))
        y = xy[-1]

        #######################
        # AE: Predict & Train #
        #######################

        pred_class = lookahead_pred_class[t]
        class_result.append(pred_class)


        if params_env['strategy'] is 'mulclass' and params_env['method'] is 'clf':
            if t < params_env['update_time']:
                logit = logits_output[t]
                if params_env['ae'] is not 'noembed':
                    pro = all_pro[t]

                reconloss=recon_loss[t]
            else:
                retrain_number=t//params_env['update_time']
                if t-retrain_number*params_env['update_time']<len(logits_output):
                    logit = logits_output[t-retrain_number*params_env['update_time']]
                    if params_env['ae'] is not 'noembed':
                        pro = all_pro[t - retrain_number * params_env['update_time']]
                    reconloss = recon_loss[t - retrain_number * params_env['update_time']]

        # pro_result.append(pro)
        if params_env['strategy'] == 'mulclass':
            seed_value = 42
            np.random.seed(seed_value)
            tf.random.set_seed(seed_value)
            random.seed(seed_value)

            ##incre learning

            if t != 0 and t != params_env['time_steps'] - 1 and (t + 1) % params_env['update_time'] == 0:
                if params_env['method'] == 'iforest':
                    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in mov_np_retrain]
                    mov_np_retrain_arr = np.concatenate(q_lst, axis=0)
                    train_iforest(mov_np_retrain_arr, iforest)
                    lookahead_pred_class = lookahead_predictions_iforest(t + 1, params_env, iforest)
                elif params_env['method'] == 'lof':
                    if len(mov_np_retrain)>0:
                        q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in mov_np_retrain]
                        mov_np_retrain_arr = np.concatenate(q_lst, axis=0)
                        train_lof(mov_np_retrain_arr, lof)
                    lookahead_pred_class = lookahead_predictions_lof(t + 1, params_env, lof)

                elif params_env['method'] == 'clf':
                    if 'vicon' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 10
                        latent_dim = 2
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001
                    if 'shuttle' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 9
                        latent_dim = 2
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001
                    if 'kdd' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 116
                        latent_dim = 20
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001
                    if 'mniscon' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 784
                        latent_dim = 20
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001
                    if 'forescon' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 52
                        latent_dim = 20
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001
                    if 'sensorless' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 48
                        latent_dim = 20
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001

                    if 'sea_con' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 2
                        latent_dim = 2
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001
                    if 'blob' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 3
                        latent_dim = 2
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001

                    if 'water' in params_env['data_source']:
                        batch_size = 32
                        input_dim = 1
                        latent_dim = 2
                        num_classes = len(ref_win_aps)+1
                        num_epochs = 10
                        lr = 0.001


                    print(t, 'to retrain')

                    retrain_time.append(t)

                    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in mov_np_retrain]
                    mov_np_retrain_arr = np.concatenate(q_lst, axis=0)


                    # Iterate through all ref_win_aps and mov_win_aps for corresponding anomalous classes
                    for i in range(class_an):
                        # Get corresponding mov_win_ap and ref_win_ap for anomalous class
                        mov_win_ap = mov_win_aps[i]
                        ref_win_ap = ref_win_aps[i]

                        # When mov_win_ap length reaches 30, concatenate and prepare for training
                        if len(mov_win_ap) == 30:
                            # Reshape all samples to ensure correct shape when concatenating
                            q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in mov_win_ap]
                            mov_win_arr = np.concatenate(q_lst, axis=0)

                            # Update ref_win_arr_ap, representing reference window for current class
                            ref_win_arr_ap = mov_win_arr
                            ref_win_ap.clear()  # Clear current window for new data
                            ref_win_ap.extend(mov_win_ap)  # Update ref_win_ap content
                            # Extend ref_win_aps content
                            ref_win_aps[i] = ref_win_ap
                            # Update mov_win_aps content
                            mov_win_aps[i] = mov_win_ap

                    if len(mov_np_retrain_arr) == 1000:
                        # Initialize X_train list, add mov_np_retrain_arr as first element
                        X_train = [mov_np_retrain_arr]
                        y_train = [0] * 1000  # Initialize y_train with 1000 normal class labels

                        # Iterate through all anomalous classes
                        for i in range(len(ref_win_aps)):
                            ref_win_arr_ap = ref_win_aps[i]
                            ref_win_arr_ap=[arr.flatten() for arr in ref_win_arr_ap]
                            if len(ref_win_arr_ap) == 30:
                                X_train.append(ref_win_arr_ap)  # Add each anomalous class ref_win_arr_ap to X_train
                                y_train.extend([i + 1] * 30)  # Update y_train, add corresponding anomalous class labels

                        X_train = np.vstack(X_train)
                        y_train = np.array(y_train)
                        print(np.unique(y_train))
                        X_train = torch.tensor(X_train, dtype=torch.float32)

                        # Print data shapes



                        # Call retraining function
                        if params_env['ae'] is not 'baseline':
                            print(t, np.shape(X_train))
                            ae_model, mlp= retrain_aemlp(ae_model, mlp, X_train, y_train, num_epochs, lr, params_env)

                        # Get new predictions and reconstruction error thresholds

                        if params_env['ae'] == 'noembed':
                            _,_,recon_loss_thre=predict_with_ae_mlp_noembed(ae_model, mlp, X_train,params_env)
                            lookahead_pred_class, logits_output, recon_loss = lookahead_predictions_aemlp_noembed(t + 1,params_env,ae_model, mlp)

                        else:
                            _, _, recon_loss_thre, _,_ = predict_with_ae_mlp(ae_model, mlp, X_train, params_env)
                            lookahead_pred_class, logits_output, recon_loss, all_pro = lookahead_predictions_aemlp(t + 1,params_env,ae_model, mlp)

                        print('retrain at time', t)

                        # Update thresholds for all classes
                        threshold_mulclass = []

                        threshold_mulclass.append(np.max(recon_loss_thre[:1000]))  # Normal class threshold

                        for i in range(len(ref_win_aps)):
                            threshold = np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                            threshold_mulclass.append(threshold)


                        print(t, 'new thresholds', threshold_mulclass)


                    else:
                        # If training conditions not met yet, continue using existing model for predictions
                        if params_env['ae'] == 'noembed':
                            lookahead_pred_class, logits_output, recon_loss = lookahead_predictions_aemlp_noembed(t + 1,params_env,ae_model, mlp)
                            print(np.max(lookahead_pred_class),'max')
                        else:
                            lookahead_pred_class, logits_output, recon_loss, all_pro = lookahead_predictions_aemlp(t + 1,params_env,ae_model, mlp)

            for i in range(len(ref_win_aps)):
                if len(ref_win_aps[i]) == 30:
                    ref_win_aps[i] = list(ref_win_aps[i])

            # Detect anomalous sequences

            if pred_class == 0:
                normal_win.append(x)
                if len(mov_win_np_mul) == 30 and len(ref_win_np_mul) == 30:
                    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in ref_win_np_mul]
                    ref_win_arr_np = np.concatenate(q_lst, axis=0)
                    distances = np.linalg.norm(ref_win_arr_np - x, axis=1)
                    euclidean_point0 = np.mean(distances)

                    if params_env['method'] == 'lof' or params_env['method'] == 'iforest':
                        if euclidean_point0 > euclidean_thresholds[0]:
                            print(t, 'normal to novel', euclidean_point0, euclidean_thresholds)
                            pred_class =class_an + 1

                            if pred_class == class_an + 1:
                                if new_first == 0:
                                    mov_win_aps.append(deque(maxlen=30))
                                    print(len(mov_win_aps), 'mov_win_aps length')
                                t_as_list[pred_class - 1].append(t)
                                mov_win_aps[pred_class - 1].append(x)
                                new_first += 1

                                ####New class exceeds 30, retrain, new window
                                if len(mov_win_aps[len(ref_lof_aps)]) == 30 and are_nearly_continuous(t_as_list[pred_class - 1]):
                                    new_first = 0
                                    new_x = mov_win_aps[len(ref_lof_aps)]
                                    ref_lof_aps.append(new_x)
                                    points = mov_win_aps[pred_class - 1]
                                    max_distance = 0
                                    for point1, point2 in combinations(points, 2):
                                        distance = np.linalg.norm(point1 - point2)
                                        if distance > max_distance:
                                            max_distance = distance
                                    euclidean_thresholds.append(1*max_distance)
                                    num_classes = pred_class + 1
                                    class_an += 1
                                    print(t, class_an, 'new class',euclidean_thresholds)

                    if params_env['method'] == 'clf':
                        if params_env['ae'] == 'aemlp' or params_env['ae'] == 'noembed' :
                            if reconloss > threshold_mulclass[0]:
                                pred_class = class_an + 1
                                print('normal to novel at', t, reconloss, threshold_mulclass[0])

                        if pred_class == class_an + 1:
                            if new_first == 0:
                                #####add correction mechanism


                                mov_win_aps.append(deque(maxlen=30))
                                print(len(mov_win_aps), 'mov_win_aps length')
                            t_as_list[pred_class - 1].append(t)
                            mov_win_aps[pred_class - 1].append(x)
                            new_first += 1

                            ####New class exceeds 30, retrain, new window
                            if len(mov_win_aps[len(ref_win_aps)]) == 30 and are_nearly_continuous(t_as_list[pred_class - 1]):
                                new_first = 0
                                # New anomalous class, create new window
                                new_x = mov_win_aps[len(ref_win_aps)]
                                ref_win_aps.append(new_x)
                                num_classes = len(ref_win_aps) + 1
                                print('new # class', num_classes)

                                X_train = [mov_np_retrain_arr]
                                y_train = [0] * 1000  # Initialize y_train with 1000 normal class labels

                                # Iterate through all anomalous classes
                                for i in range(len(ref_win_aps)):
                                    ref_win_arr_ap = ref_win_aps[i]
                                    ref_win_arr_ap = [arr.flatten() for arr in ref_win_arr_ap]
                                    if len(ref_win_arr_ap) == 30:
                                        X_train.append(ref_win_arr_ap)  # Add each anomalous class ref_win_arr_ap to X_train
                                        y_train.extend([i + 1] * 30)  # Update y_train, add corresponding anomalous class labels

                                # Convert X_train to numpy array and vertically stack

                                X_train = np.vstack(X_train)
                                y_train = np.array(y_train)

                                ###upsampling
                                arr = X_train
                                majority_class = arr[:1000]

                                minority_classes = []
                                start_index = 1000
                                while start_index < len(X_train):
                                    end_index = start_index + 30
                                    minority_classes.append(X_train[start_index:end_index])
                                    start_index = end_index

                                # Oversample each minority class
                                print('minority class', len(minority_classes))
                                majority_class_size = len(majority_class)

                                num_minorities = len(minority_classes)
                                # smote
                                class_counts = Counter(y_train)
                                majority_class_label = class_counts.most_common(1)[0][0]  # Majority class label
                                majority_class_size = class_counts[majority_class_label]

                                ###############
                                #correction in latent space
                                if params_env['correc'] == 'true':

                                    ######
                                    X_resampled, y_resampled= correct_and_resample_in_input_keep_nearest_minority_density_adaptive_new(
                                        X_train,
                                        y_train,
                                        standardize_X=True,  # Whether to standardize (recommended)
                                        distance="mahalanobis",  # Use :contentReference[oaicite:0]{index=0}
                                        do_smote=True,  # Enable :contentReference[oaicite:1]{index=1} oversampling
                                        verbose=True
                                    )
                                    ##


                                    print("Original data:", X_train.shape, y_train.shape)
                                    print("After resampling:", X_resampled.shape, y_resampled.shape)


                                ###########
                                else:
                                    smote = SMOTE(sampling_strategy='auto', random_state=42)
                                    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)



                                X_train= X_resampled
                                y_train = y_resampled
                                # Check class distribution after oversampling
                                print("Resampled dataset shape:", Counter(y_resampled))

                                if 'noupsampling' in params_env['ind']:
                                    X_train = np.vstack([majority_class] + minority_classes)
                                    y_train = np.array(
                                        [0] * 1000 + [i + 1 for i in range(num_minorities) for _ in range(30)]
                                        # Each minority class has 30 samples
                                    )

                                    # Convert to PyTorch tensor
                                X_train = torch.tensor(X_train, dtype=torch.float32)


                                ###create new and retrain
                                print(t, 'new model')
                                print(X_train,y_train)
                                ae_model, mlp = train_ae_mlp(params_env,X_train, y_train, input_dim, latent_dim, num_classes,
                                                             num_epochs=20, lr=lr)

                                # If training conditions not met yet, continue using existing model for predictions
                                if params_env['ae'] == 'noembed':
                                    _, _, recon_loss_thre = predict_with_ae_mlp_noembed(ae_model, mlp, X_train,params_env)
                                    lookahead_pred_class, logits_output, recon_loss = lookahead_predictions_aemlp_noembed(t + 1, params_env, ae_model, mlp)
                                else:
                                    _, _, recon_loss_thre, _,_ = predict_with_ae_mlp(ae_model, mlp, X_train, params_env)
                                    lookahead_pred_class, logits_output, recon_loss, all_pro = lookahead_predictions_aemlp(t + 1, params_env, ae_model, mlp)

                                retrain_time.append(t)
                                class_an+=1
                                # print(all_pro,np.shape(all_pro))
                                print('retrained, # of anomalous classes',class_an)

                                # Update thresholds for all classes
                                threshold_mulclass = []
                                threshold_mulclass.append(np.max(recon_loss_thre[:1000]))
                                for i in range(len(ref_win_aps)):
                                    threshold = np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                                    threshold_mulclass.append(threshold)

                                print(t, 'new thresholds', threshold_mulclass)


                if pred_class == 0:
                    if params_env['method'] == 'clf':
                        recon_mov.append(reconloss)
                        if len(recon_mov) > 30:
                            recon_mov.pop(0)
                        if len(recon_ref) < 30:
                            recon_ref = np.append(recon_ref, reconloss)

                    if len(ref_win_np_mul) < 30:
                        if len(ref_win_np_mul) == 0:
                            print('ref win np start')
                        ref_win_np_mul.append(x)

                    if len(mov_win_np_mul) == 0:
                        print('mov win np start')
                    mov_win_np_mul.append(x)
                    mov_np_retrain.append(x)

            if pred_class > 0 and pred_class <= class_an:
                t_ap = t
                euclidean_points=[]
                if params_env['method'] == 'lof' or params_env['method'] == 'iforest':
                    for i in range(len(ref_lof_aps)):
                        distances = np.linalg.norm(ref_lof_aps[i] - x, axis=1)
                        euclidean_point = np.mean(distances)
                        euclidean_points.append(euclidean_point)
                    min_distance = min(euclidean_points)
                    min_index = euclidean_points.index(min_distance)
                    pred_class = min_index + 1
                # Check if this distance exceeds corresponding threshold
                    if min_distance > euclidean_thresholds[pred_class]:
                        print(t, 'ano to novel', pred_class, min_distance, euclidean_thresholds)
                        pred_class = class_an + 1
                    euclidean_points = []

                    if pred_class == class_an + 1:
                        if new_first==0:
                            mov_win_aps.append(deque(maxlen=30))
                        t_as_list[pred_class-1].append(t)
                        mov_win_aps[pred_class-1].append(x)
                        new_first+=1
                    ####New class exceeds 30, retrain, new window
                        if len(mov_win_aps[len(ref_lof_aps)]) == 30 and are_nearly_continuous(t_as_list[pred_class - 1]):
                            new_first = 0
                            points = mov_win_aps[pred_class - 1]
                            max_distance = 0
                            new_x = mov_win_aps[pred_class - 1]
                            ref_lof_aps.append(new_x)
                            for point1, point2 in combinations(points, 2):
                                distance = np.linalg.norm(point1 - point2)
                                if distance > max_distance:
                                    max_distance = distance
                            euclidean_thresholds.append( 1*max_distance)
                            num_classes = pred_class + 1
                            class_an += 1
                            print(t,class_an,'new class',euclidean_thresholds)

                    if pred_class > 0 and pred_class <= len(ref_win_aps):
                        # print(t, pred_class, t_as_list, 't_as_list')
                        mov_win_aps[pred_class - 1].append(x)
                        t_as_list[pred_class - 1].append(t)

                if params_env['method'] == 'clf':
                    if params_env['ae'] == 'aemlp' or params_env['ae'] == 'noembed' :
                        if reconloss > threshold_mulclass[pred_class]:
                            print('ano to novel', pred_class,t, reconloss, threshold_mulclass[pred_class])
                            pred_class = class_an + 1

                    if pred_class == class_an + 1:
                        if new_first==0:
                            mov_win_aps.append(deque(maxlen=30))
                            print(len(mov_win_aps),'mov_win_aps length')
                        t_as_list[pred_class-1].append(t)
                        mov_win_aps[pred_class-1].append(x)
                        new_first+=1
                        # print( 'new class',t, pred_class,reconloss, threshold_mulclass)

                        ####New class exceeds 30, retrain, new window
                        if len(mov_win_aps[len(ref_win_aps)]) == 30 and are_nearly_continuous(t_as_list[pred_class - 1]):
                            new_first = 0
                            # New anomalous class, create new window
                            new_x = mov_win_aps[len(ref_win_aps)]
                            ref_win_aps.append(new_x)
                            num_classes = len(ref_win_aps) + 1
                            print('new # class', num_classes)

                            X_train = [mov_np_retrain_arr]
                            y_train = [0] * 1000  # Initialize y_train with 1000 normal class labels

                            # Iterate through all anomalous classes
                            for i in range(len(ref_win_aps)):
                                ref_win_arr_ap = ref_win_aps[i]
                                ref_win_arr_ap=[arr.flatten() for arr in ref_win_arr_ap]
                                if len(ref_win_arr_ap) == 30:
                                    X_train.append(ref_win_arr_ap)  # Add each anomalous class ref_win_arr_ap to X_train
                                    y_train.extend([i + 1] * 30)  # Update y_train, add corresponding anomalous class labels

                            # Convert X_train to numpy array and vertically stack

                            X_train = np.vstack(X_train)
                            y_train = np.array(y_train)

                            ###upsampling
                            arr = X_train
                            majority_class = arr[:1000]

                            minority_classes = []
                            start_index = 1000
                            while start_index < len(X_train):
                                end_index = start_index + 30
                                minority_classes.append(X_train[start_index:end_index])
                                start_index = end_index

                            # Oversample each minority class
                            print('minority class', len(minority_classes))

                            num_minorities = len(minority_classes)
                            # smote
                            class_counts = Counter(y_train)
                            majority_class_label = class_counts.most_common(1)[0][0]  # Majority class label
                            majority_class_size = class_counts[majority_class_label]
                            # Initialize SMOTE
                            smote = SMOTE(sampling_strategy='auto', random_state=42)
                            # Use SMOTE for oversampling, making minority class sample count equal to majority class
                            ###Only sample core points, not touching boundary/outlier points
                            ############
                            # if params_env['correc'] == 'true':
                            #     # Print original class distribution
                            #     original_classes, original_counts = np.unique(y_train, return_counts=True)
                            #     print("\nOriginal class distribution:")
                            #     for cls, count in zip(original_classes, original_counts):
                            #         print(f"Class {cls}: {count} samples")
                            #
                            #     classes = np.unique(y_train)
                            #     class_counts = np.bincount(y_train)
                            #     majority_class = np.argmax(class_counts)
                            #     X_majority = X_train[y_train == majority_class]
                            #
                            #     # Initialize processed data
                            #     X_processed = X_majority.copy()
                            #     y_processed = np.full(len(X_majority), majority_class)
                            #
                            #     # Noise ratio threshold (exceed this value then keep all samples)
                            #     noise_threshold = 0.7  # 50%
                            #
                            #     for cls in classes:
                            #         if cls == majority_class or class_counts[cls] == 0:
                            #             continue
                            #
                            #         X_min = X_train[y_train == cls]
                            #         print(f"\nProcessing class {cls} (original sample count: {len(X_min)})")
                            #
                            #         # If too few samples, keep directly
                            #         if len(X_min) <= 5:
                            #             X_min_core = X_min
                            #             print(f" - Sample count≤5, keep all")
                            #         else:
                            #             # DBSCAN clustering
                            #             #print variation of each class before correction:t,cls,var(X_min)
                            #
                            #             print(f"\nProcessing class {cls} (original sample count: {len(X_min)})")
                            #             var_per_dim = np.var(X_min, axis=0)  # Variance per dimension
                            #             mean_var = np.mean(var_per_dim)  # Average variance (overall "dispersion")
                            #             print(f"Class {cls} per-dimension variance: {var_per_dim}, average variance: {mean_var:.4f}")
                            #
                            #             db = DBSCAN(eps=0.5, min_samples=5).fit(X_min)
                            #             core_mask = db.labels_ != -1
                            #             noise_mask = db.labels_ == -1
                            #             noise_ratio = np.sum(noise_mask) / len(X_min)
                            #
                            #             # If noise ratio too high, keep all samples
                            #             if noise_ratio > noise_threshold:
                            #                 X_min_core = X_min
                            #                 print(f" - Noise ratio too high ({noise_ratio:.1%}), keep all")
                            #             else:
                            #                 X_min_core = X_min[core_mask]
                            #                 print(f" - Keep core points: {len(X_min_core)} samples, noise: {np.sum(noise_mask)}")
                            #
                            #
                            #             var_per_dim_core = np.var(X_min_core, axis=0)  # Variance per dimension
                            #             mean_var = np.mean(var_per_dim_core)  # Average variance (overall "dispersion")
                            #             print(f"After correction, class {cls} per-dimension variance: {var_per_dim_core}, average variance: {mean_var:.4f}")
                            #         # Add to dataset
                            #         X_processed = np.vstack([X_processed, X_min_core])
                            #         y_processed = np.hstack([y_processed, np.full(len(X_min_core), cls)])
                            #
                            #     # Check number of classes
                            #     if len(np.unique(y_processed)) < 2:
                            #         print("\nWarning: Only 1 class remaining after processing, skipping SMOTE")
                            #         X_resampled, y_resampled = X_processed, y_processed
                            #     else:
                            #         smote = SMOTE()
                            #         X_res, y_res = smote.fit_resample(X_processed, y_processed)
                            #         y_res[y_res == -1] = majority_class  # Handle possible noise labels
                            #         X_resampled, y_resampled = X_res, y_res
                            #
                            #     # Print final distribution
                            #     print("\nFinal class distribution:")
                            #     resampled_classes, resampled_counts = np.unique(y_resampled, return_counts=True)
                            #     for cls, count in zip(resampled_classes, resampled_counts):
                            #         print(f"Class {cls}: {count} samples")

                            ###########
                            if params_env['correc'] == 'true':
                                # Correction + oversampling in embedding space, then decode back to input space
                                # X_resampled, y_resampled = correct_and_resample_in_embedding(
                                #     ae_model,
                                #     X_train,  # Here X_train is numpy (you already have np.vstack array above)
                                #     y_train,
                                #     noise_threshold=0.9,
                                #     dbscan_eps=eps  ,
                                #     dbscan_min_samples=min_samples,
                                #     do_smote=True,
                                #     device="cpu"
                                # )
                                # First ensure you have imported the function definitions above

                                # X_resampled, y_resampled = correct_and_resample_in_embedding_keep_nearest(
                                #     ae_model,
                                #     X_train_np=X_train,
                                #     y_train_np=y_train,
                                #     keep_ratio=0.95,  # Keep closest 95% per class
                                #     distance="mahalanobis",  # Can also write "euclidean"
                                #     standardize_Z=True,  # Whether to standardize Z (recommended)
                                #     do_smote=True,  # Whether to perform SMOTE in input space
                                #     smote_kwargs=None,  # SMOTE parameters (optional)
                                #     device="cpu",
                                #     verbose=True  # Print retention counts and thresholds
                                # )

                                # X_resampled, y_resampled = correct_and_resample_in_embedding_keep_nearest_minority_density_adaptive(
                                #     t,
                                #     ae_model,
                                #     X_train_np=X_train,
                                #     y_train_np=y_train,
                                #     keep_min=0.9, keep_max=0.99,  # Retention ratio upper/lower bounds
                                #     knn_k=10,  # k for density estimation
                                #     distance="mahalanobis",  # Sorting distance
                                #     standardize_Z=True,
                                #     do_smote=True,
                                #     smote_kwargs={'k_neighbors': 5},
                                #     device="cpu",
                                #     verbose=True,
                                #     min_keep_per_class=5
                                # )
                                # X_resampled, y_resampled = correct_and_resample_in_input_keep_nearest_minority_density_adaptive(
                                #     X_train_np=X_train,
                                #     y_train_np=y_train,
                                #     keep_min=0.85, keep_max=0.99,
                                #     knn_k=20,
                                #     distance="mahalanobis",  # or "euclidean"
                                #     standardize_X=True,  # Strongly recommend True
                                #     do_smote=True,
                                #     smote_kwargs={'k_neighbors': 5, 'random_state': 42},
                                #     verbose=True,
                                #     min_keep_per_class=5
                                # )
                                X_resampled, y_resampled= correct_and_resample_in_input_keep_nearest_minority_density_adaptive_new(
                                    X_train,
                                    y_train,
                                    standardize_X=True,  # Whether to standardize (recommended)
                                    distance="mahalanobis",  # Use :contentReference[oaicite:0]{index=0}
                                    do_smote=True,  # Enable :contentReference[oaicite:1]{index=1} oversampling
                                    verbose=True
                                )

                                # X_resampled, y_resampled = correct_and_resample_X_via_embedding_new(
                                #     X_train,
                                #     y_train,
                                #     ae_model,
                                #     device="cpu",  # If model on GPU, use "cuda", otherwise "cpu"
                                #     keep_min=0.85,  # Minimum retention ratio
                                #     keep_max=0.99,  # Maximum retention ratio
                                #     knn_k=10,  # kNN for density estimation
                                #     distance="mahalanobis",  # Use Mahalanobis distance for filtering
                                #     do_smote=True,  # Whether to enable SMOTE oversampling
                                #     decode_after_smote=True  # Need decoder to decode embedding synthetic samples back to X
                                # )

                                print("Original data:", X_train.shape, y_train.shape)
                                print("After resampling:", X_resampled.shape, y_resampled.shape)



                            else:
                                # Keep your original branch
                                smote = SMOTE(sampling_strategy='auto', random_state=42)
                                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                            X_train = X_resampled
                            y_train = y_resampled
                            # Check class distribution after oversampling
                            # print("Resampled dataset shape:", Counter(y_resampled))

                            X_train = torch.tensor(X_train, dtype=torch.float32)

                            ###create new and retrain
                            print(t,'new model',len(ref_win_aps))
                            ae_model, mlp = train_ae_mlp(params_env,X_train, y_train,input_dim, latent_dim, num_classes,
                                                         num_epochs=20, lr=lr)

                            # If training conditions not met yet, continue using existing model for predictions
                            if params_env['ae'] == 'noembed':
                                _, _, recon_loss_thre = predict_with_ae_mlp_noembed(ae_model, mlp, X_train, params_env)
                                lookahead_pred_class, logits_output, recon_loss = lookahead_predictions_aemlp_noembed(t + 1,params_env, ae_model, mlp)
                            else:
                                _, _, recon_loss_thre, _,_ = predict_with_ae_mlp(ae_model, mlp, X_train, params_env)
                                lookahead_pred_class, logits_output, recon_loss, all_pro = lookahead_predictions_aemlp(t + 1, params_env, ae_model, mlp)


                            retrain_time.append(t)
                            class_an+=1
                            print('retrained, # of anomalous classes', class_an)

                            # Update thresholds for all classes
                            threshold_mulclass = []

                            if 'kdd' in params_env['data_source']:
                                threshold_mulclass.append(10 * np.max(recon_loss_thre[:1000]))  # Normal class threshold

                                for i in range(len(ref_win_aps)):
                                    threshold = np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                                    threshold_mulclass.append(threshold)
                            elif 'mniscon' in params_env['data_source']:
                                threshold_mulclass.append(2*np.max(recon_loss_thre[:1000]))  # Normal class threshold
                                for i in range(len(ref_win_aps)):
                                    threshold =50 * np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                                    threshold_mulclass.append(threshold)

                            elif 'blob' in params_env['data_source'] or 'water' in params_env['data_source'] :
                                threshold_mulclass.append(5* np.max(recon_loss_thre[:1000]))  # Normal class threshold

                                for i in range(len(ref_win_aps)):
                                    # threshold = 5*np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                                    threshold = 100 * np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                                    threshold_mulclass.append(threshold)
                            else:
                                threshold_mulclass.append(np.max(recon_loss_thre[:1000]))  # Normal class threshold
                                for i in range(len(ref_win_aps)):
                                    threshold = np.max(recon_loss_thre[1000 + i * 30: 1030 + i * 30])
                                    threshold_mulclass.append(threshold)

                                print(t, 'new thresholds', threshold_mulclass)

                    # Update corresponding class window
                    if pred_class > 0 and pred_class <= len(ref_win_aps):
                        # print(t, pred_class, t_as_list, 't_as_list')
                        mov_win_aps[pred_class - 1].append(x)
                        t_as_list[pred_class - 1].append(t)


            if pred_class != 0:
                print(t, pred_class)




        ###############
        # Correctness #
        ###############
        correct = 1 if y == pred_class else 0  # check if prediction was correct

        if y==0:
            preq_specificity_s, preq_specificity_n, preq_specificity = update_preq_metric(preq_specificity_s,
                                                                                          preq_specificity_n, correct,
                                                                                          params_env['preq_fading_factor'])
        else:
            preq_recall_s, preq_recall_n, preq_recall = update_preq_metric(preq_recall_s, preq_recall_n, correct,
                                                                           params_env['preq_fading_factor'])

        preq_recalls[t] = preq_recall
        preq_specificities[t] = preq_specificity
        ########################
        # Update preq. metrics #
        ########################

        # update general accuracy
        preq_general_acc_s, preq_general_acc_n, preq_general_acc = \
            update_preq_metric(preq_general_acc_s, preq_general_acc_n, correct, params_env['preq_fading_factor'])
        preq_general_accs.append(preq_general_acc)

        # update class accuracies & gmean

        preq_class_acc_s[y], preq_class_acc_n[y], preq_class_acc[y] = update_preq_metric(
            preq_class_acc_s[y], preq_class_acc_n[y], correct, params_env['preq_fading_factor'])

        lst = []
        for k, v in preq_class_acc.items():
            preq_class_accs[k].append(v)
            lst.append(v)


        gmean = np.power(np.prod(lst), 1.0 / len(lst))

        preq_gmeans.append(gmean)
        # print(preq_gmeans[t], 'g_means', t)




    return preq_class_accs, preq_gmeans, preq_recalls,preq_specificities,retrain_time,t_as_list,class_result


# datasets=['blob_drift','sea_con_drift','forescon_drift','kdd_drift','water_drift']
datasets=['shuttle']
gmean_final=[]
pred_final=[]
for ds in datasets:
    lst_ds = [
        {'repeats': 1, 'data_source': ds,
         'strategy': 'mulclass',
         'method': 'iforest',
         'unsupervised_win_size_update': 1.0,
         'unsupervised_win_size': 1000,
         'com':'28',
         # 'ae':'baseline',
         'ae': 'aemlp',
         # 'ae': 'noembed',

         'ind':'mulincr',
         # 'ind': 'noupsampling',
         'correc':'true',
         'adaptive': 'yes', 'num_epochs': 100}
        ]
    params_env = lst_ds[0]
    for r in range(params_env['repeats']):
        print('Repetition: ', r)
        start_time = time.time()
        preq_class_accs, preq_gmeans, preq_recalls,preq_specificities,retrain_time,t_as_list,class_result=run_con(params_env)
        end_time = time.time()
        duration = end_time - start_time
        print(f"run_con repetition {r} runtime: {duration:.2f} seconds")
        gmean_final.append(preq_gmeans)
        pred_final.append(class_result)
    out_dir = 'exps/'
    filename= os.path.join(os.getcwd(), out_dir, str(params_env['strategy']) +'_'+ str(params_env['method'])+'_'+ str(params_env['data_source']) + '_' + str(params_env['ae']) + '_' + str(
                                      params_env['ind'])+str(params_env['com'])+str(params_env['correc'])+ '_preq_gmean.txt')
    # Only use when gmean_final is multi-nested
    gmean_final2 = np.array([np.array(row).flatten() for row in gmean_final], dtype=float)
    np.savetxt(filename, gmean_final2, delimiter=', ', fmt='%1.6f')

    filename = f"{params_env['data_source']}_{params_env['ae']}_{params_env['method']}_{params_env['correc']}_pred.txt"
    with open(filename, 'w') as f:
        for item in pred_final:
            f.write(f"{item}\n")