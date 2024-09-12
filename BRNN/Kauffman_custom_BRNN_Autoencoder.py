# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Libraries

# !pip install -r requirements_BRNN.txt

# +
import os
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import random

try:
    import wandb
except:
    # !python3 -m pip install wandb
    import wandb

import warnings
warnings.filterwarnings('ignore')
torch.cuda.device_count()
# -

# ## GPU/CPU Devices

# +
if torch.cuda.is_available():
    print("CUDA is available. Here are the GPU details:")
    # List all available GPUs.
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")
    

print("Available CPUs:")
print(f"CPU count: {torch.get_num_threads()}")

# +
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
# -

# ## Log File

# +
counter = 1
base_directory = f"./BRRN_runs/BRRN_run_{counter}"

# Check existing directories and increment to find the next available 'run_x' directory
while os.path.exists(base_directory):
    counter += 1
    base_directory = f"./BRRN_runs/BRRN_run_{counter}"

# Create the new run directory
os.makedirs(base_directory)
print(f"New run directory created: {base_directory}")
# -

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S %Z")
current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
f.write(f"\n------------------------------------------\n\nBRRN run\nTime: {current_time}\nDate: {current_date}\n")
f.close()

# ## Loading data

r=0
redshifts, o3lum,o3corr, bpt1,bpt2, rml50, rmu, con, d4n,hda, vdisp = [],[],[],[],[],[],[],[],[],[],[]
with open("data/agn.dat_dr4_release.v2", 'r') as file:
    for line in file:
        parts = line.split()  # Splits the line into parts
        redshifts.append(float(parts[5]))
        o3lum.append(float(parts[6]))
        o3corr.append(float(parts[7]))
        bpt1.append(float(parts[8]))
        bpt2.append(float(parts[9]))
        rml50.append(float(parts[10]))
        rmu.append(float(parts[11]))
        con.append(float(parts[12]))
        d4n.append(float(parts[13]))
        hda.append(float(parts[14]))
        vdisp.append(float(parts[15]))
        r+=1
redshifts, o3lum,o3corr, bpt1,bpt2, rml50, rmu, con, d4n,hda, vdisp = np.array(redshifts), np.array(o3lum),np.array(o3corr), np.array(bpt1),np.array(bpt2), np.array(rml50), np.array(rmu), np.array(con), np.array(d4n),np.array(hda), np.array(vdisp)

df_lc = pd.read_parquet('data/df_lc_kauffmann.parquet')

bands_inlc = ['zg', 'zr', 'zi', 'W1', 'W2']
colors = ['b', 'g', 'orange', 'c', 'm']

# ## Data Preprocessing

# ### Obtain maximum coverages

# +
# Store objectid
max_oid_per_band = {}
# Store coverage
max_time_coverage_per_band = {}
# Store Max lengths
max_length_per_band = {}

for band in bands_inlc:
    band_data = df_lc.xs(band, level='band')
    max_oid = band_data.groupby('objectid').size().idxmax()
    max_oid_per_band[band] = max_oid
    max_time_coverage_per_band[band] = band_data.xs(max_oid, level='objectid').index.get_level_values('time').unique()
    max_length_per_band[band] = len(max_time_coverage_per_band[band])

print("Maximum Lenghts Coverage Per Band:")
for band, times_len in max_length_per_band.items():
    print(f"Band {band}: {times_len}")

# +
num_bands = len(bands_inlc)

fig, axes = plt.subplots(nrows=num_bands, figsize=(14, 10), sharex=True)

for i, (band, ax) in enumerate(zip(bands_inlc, axes.flatten())):
    oid = max_oid_per_band[band]
    data = df_lc.xs((oid, band), level=('objectid', 'band'))
    if not data.empty:
        ax.plot(data.index.get_level_values('time'), data['flux'], marker='o', linestyle='-', color=colors[i], label=f'Flux for AGN {oid}', alpha=0.7)

    ax.set_title(f'Flux vs. Time for Band {band}')
    ax.set_ylabel('Flux')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})

axes[-1].set_xlabel('Time')
fig_title = 'Maximum Time Coverage'
fig.suptitle(fig_title)
plt.tight_layout()
plt.show()

# Save the figure
if not os.path.exists(os.path.join(base_directory,'output')):
    os.makedirs(os.path.join(base_directory,'output'))
save_path = os.path.join(os.path.join(base_directory,'output'), fig_title.replace(' ', '_') + '.png')
fig.savefig(save_path)
print(f"Saved figure to {save_path}")
# -

# # Multivariate BRNN

# ## Data preprocessing

# #### Normalization

# +
time_scaler = MinMaxScaler()
flux_scaler = MinMaxScaler()

filter_bands = df_lc.index.get_level_values('band').isin(bands_inlc)

times = df_lc.index.get_level_values('time')[filter_bands].values.reshape(-1, 1)
fluxes = df_lc['flux'][filter_bands].values.reshape(-1, 1)
# -

print(f'Minimum times value: {times.min()}')
print(f'Maximum times value: {times.max()}')
print(f'Minimum flux value: {fluxes.min()}')
print(f'Maximum flux value: {fluxes.max()}')

normalized_times = time_scaler.fit(times)
normalized_fluxes = flux_scaler.fit(fluxes)

normalized_times = time_scaler.transform(df_lc.index.get_level_values('time').values.reshape(-1, 1))
normalized_fluxes = flux_scaler.transform(df_lc['flux'].values.reshape(-1, 1))

df_lc['time_norm'] = normalized_times.flatten()

df_lc['flux_norm'] = normalized_fluxes.flatten()

# #### Take a subset

# +
# Select the first subset_size object IDs
subset_size = 80000
obj_ids_subset = df_lc.index.get_level_values('objectid').unique()[:subset_size]
#obj_ids_subset = df_lc.index.get_level_values('objectid').unique()

# Extract the data for the selected objects
df_lc_subset = df_lc.loc[obj_ids_subset]
redshifts_subset = {obj_id: redshifts[obj_id] for obj_id in obj_ids_subset}
# -

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\nNumber of AGN in the subset: {subset_size}\n")
f.close()


# #### Fill the sequences with padding

# ##### Without normalization

def unify_lc_for_rnn_multi_band(df_lc, redshifts, max_length_per_band, bands_inlc=['zg', 'zr', 'zi', 'W1', 'W2'], padding_value=-1):
    objids = df_lc.index.get_level_values('objectid').unique()
    if isinstance(redshifts, np.ndarray):
        redshifts = dict(zip(objids, redshifts))
    padded_times_all, padded_fluxes_all = [], []
    
    for obj in tqdm(objids, desc="Processing objects"):
        redshift = redshifts.get(obj, None)
        if redshift is None:
            continue
        singleobj = df_lc.loc[obj]
        label = singleobj.index.unique('label')[0]
        bands = singleobj.index.get_level_values('band').unique()
        
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            obj_times, obj_fluxes = [], []
            for band in bands_inlc:
                if (label, band) in singleobj.index:
                    band_lc = singleobj.xs((label, band), level=('label', 'band'))
                    band_lc_clean = band_lc[(band_lc.index.get_level_values('time') > 56000) & (band_lc.index.get_level_values('time') < 65000)]
                    x = np.array(band_lc_clean.index.get_level_values('time'))
                    y = np.array(band_lc_clean.flux)
                    
                    sorted_indices = np.argsort(x)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    
                    if len(x) > max_length_per_band[band]:
                        x = x[:max_length_per_band[band]]
                        y = y[:max_length_per_band[band]]
                    
                    if len(x) > 0:
                        padded_x = np.pad(x, (0, max_length_per_band[band] - len(x)), 'constant', constant_values=(padding_value,))
                        padded_y = np.pad(y, (0, max_length_per_band[band] - len(y)), 'constant', constant_values=(padding_value,))
                        obj_times.extend(padded_x)
                        obj_fluxes.extend(padded_y)
                    else:
                        break
            if len(obj_times) == sum(max_length_per_band.values()):
                padded_times_all.append(obj_times)
                padded_fluxes_all.append(obj_fluxes)
    padded_times_all = np.array(padded_times_all, dtype="float32")
    padded_fluxes_all = np.array(padded_fluxes_all, dtype="float32")
    return padded_times_all, padded_fluxes_all


# ##### With normalization

def unify_lc_for_rnn_multi_band(df_lc, redshifts, max_length_per_band, bands_inlc=['zg', 'zr', 'zi', 'W1', 'W2'], padding_value=-1):
    objids = df_lc.index.get_level_values('objectid').unique()
    if isinstance(redshifts, np.ndarray):
        redshifts = dict(zip(objids, redshifts))
    padded_times_all, padded_fluxes_all = [], []
    
    for obj in tqdm(objids, desc="Processing objects"):
        redshift = redshifts.get(obj, None)
        if redshift is None:
            continue
        singleobj = df_lc.loc[obj]
        label = singleobj.index.unique('label')[0]
        bands = singleobj.index.get_level_values('band').unique()
        
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            obj_times, obj_fluxes = [], []
            for band in bands_inlc:
                if (label, band) in singleobj.index:
                    band_lc = singleobj.xs((label, band), level=('label', 'band'))
                    band_lc_clean = band_lc[(band_lc.index.get_level_values('time') > 56000) & (band_lc.index.get_level_values('time') < 65000)]
                    x = np.array(band_lc_clean.time_norm)
                    y = np.array(band_lc_clean.flux_norm)
                    
                    sorted_indices = np.argsort(x)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    
                    if len(x) > max_length_per_band[band]:
                        x = x[:max_length_per_band[band]]
                        y = y[:max_length_per_band[band]]
                    
                    if len(x) > 0:
                        padded_x = np.pad(x, (0, max_length_per_band[band] - len(x)), 'constant', constant_values=(padding_value,))
                        padded_y = np.pad(y, (0, max_length_per_band[band] - len(y)), 'constant', constant_values=(padding_value,))
                        obj_times.extend(padded_x)
                        obj_fluxes.extend(padded_y)
                    else:
                        break
            if len(obj_times) == sum(max_length_per_band.values()):
                padded_times_all.append(obj_times)
                padded_fluxes_all.append(obj_fluxes)
    padded_times_all = np.array(padded_times_all, dtype="float32")
    padded_fluxes_all = np.array(padded_fluxes_all, dtype="float32")
    return padded_times_all, padded_fluxes_all


# ##### Train and test sets

# +
from sklearn.model_selection import train_test_split

def unify_lc_for_rnn_multi_band_train_test(df_lc, redshifts, max_length_per_band, bands_inlc=['zg', 'zr', 'zi', 'W1', 'W2'], padding_value=-1, test_size=0.2):
    objids = df_lc.index.get_level_values('objectid').unique()
    if isinstance(redshifts, np.ndarray):
        redshifts = dict(zip(objids, redshifts))
    padded_times_train_all, padded_fluxes_train_all = [], []
    padded_times_test_all, padded_fluxes_test_all = [], []
    padded_test_idx_all = []
    padded_test_idx_accum = []
    
    # Calculate max lengths for train and test
    max_length_per_band_train = {}
    max_length_per_band_test = {}
    for band in bands_inlc:
        max_length_per_band_train[band] = round((1-test_size)*max_length_per_band[band])
        max_length_per_band_test[band] = max_length_per_band[band] - max_length_per_band_train[band]+1
    
    train_count = sum(max_length_per_band_train.values())
    test_count = sum(max_length_per_band_test.values())
    
    for obj in tqdm(objids, desc="Processing objects"):
        redshift = redshifts.get(obj, None)
        if redshift is None:
            continue
        singleobj = df_lc.loc[obj]
        label = singleobj.index.unique('label')[0]
        bands = singleobj.index.get_level_values('band').unique()
        
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            obj_times_train, obj_fluxes_train = [], []
            obj_times_test, obj_fluxes_test = [], []
            obj_test_idx = {}
            obj_test_idx_accum = []
            cumulative_length = 0
            for band in bands_inlc:
                if (label, band) in singleobj.index:
                    band_lc = singleobj.xs((label, band), level=('label', 'band'))
                    band_lc_clean = band_lc[(band_lc.index.get_level_values('time') > 56000) & (band_lc.index.get_level_values('time') < 65000)]
                    x = np.array(band_lc_clean.time_norm)
                    y = np.array(band_lc_clean.flux_norm)
                    
                    sorted_indices = np.argsort(x)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    
                    if len(x) > max_length_per_band[band]:
                        x = x[:max_length_per_band[band]]
                        y = y[:max_length_per_band[band]]
                    
                    
                    if len(x) > 10:
                        # Split into training and testing
                        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
                        indices = np.arange(len(x))

                        if indices.size > 0:
                            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=0)
                            # Sort the indices to ensure they are ordered
                            train_indices = np.sort(train_idx)
                            test_indices = np.sort(test_idx)
                            x_train = x[train_indices]
                            x_test = x
                            y_train = y[train_indices]
                            y_test  = y

                        # Padding for training
                        padded_x_train = np.pad(x_train, (0, max_length_per_band_train[band] - len(x_train)), 'constant', constant_values=(padding_value,))
                        padded_y_train = np.pad(y_train, (0, max_length_per_band_train[band] - len(y_train)), 'constant', constant_values=(padding_value,))
                        
                        # Padding for testing
                        padded_x_test = np.pad(x_test, (0, max_length_per_band[band] - len(x_test)), 'constant', constant_values=(padding_value,))
                        padded_y_test = np.pad(y_test, (0, max_length_per_band[band] - len(y_test)), 'constant', constant_values=(padding_value,))
                        obj_times_train.extend(padded_x_train)
                        obj_fluxes_train.extend(padded_y_train)
                        obj_times_test.extend(padded_x_test)
                        obj_fluxes_test.extend(padded_y_test)
                        # Test Indexes
                        obj_test_idx[band] = test_indices
                        obj_test_idx_accum.extend(test_indices + cumulative_length)
                        cumulative_length += max_length_per_band[band]
            if len(obj_times_train) == train_count: # and len(obj_times_test) == test_count:
                padded_times_train_all.append(obj_times_train)
                padded_fluxes_train_all.append(obj_fluxes_train)
                padded_times_test_all.append(obj_times_test)
                padded_fluxes_test_all.append(obj_fluxes_test)
                padded_test_idx_all.append(obj_test_idx)
                padded_test_idx_accum.append(obj_test_idx_accum)

    padded_times_train_all = np.array(padded_times_train_all, dtype="float32")
    padded_fluxes_train_all = np.array(padded_fluxes_train_all, dtype="float32")
    padded_times_test_all = np.array(padded_times_test_all, dtype="float32")
    padded_fluxes_test_all = np.array(padded_fluxes_test_all, dtype="float32")
    
    return padded_times_train_all, padded_fluxes_train_all, padded_times_test_all, padded_fluxes_test_all, padded_test_idx_all,padded_test_idx_accum, max_length_per_band_train, max_length_per_band_test

# +

# Use the modified function to prepare the data
padded_times_train, padded_fluxes_train, padded_times_test, padded_fluxes_test, padded_test_idx_all, padded_test_idx_accum, max_length_per_band_train, max_length_per_band_test = unify_lc_for_rnn_multi_band_train_test(df_lc_subset, redshifts_subset, max_length_per_band)

# Convert the data to PyTorch tensors
padded_times_train_tensor = torch.tensor(padded_times_train, dtype=torch.float32)
padded_fluxes_train_tensor = torch.tensor(padded_fluxes_train, dtype=torch.float32)
padded_times_test_tensor = torch.tensor(padded_times_test, dtype=torch.float32)
padded_fluxes_test_tensor = torch.tensor(padded_fluxes_test, dtype=torch.float32)

# Combine time and flux into a single input tensor
input_train_tensor = torch.stack((padded_times_train_tensor, padded_fluxes_train_tensor), dim=-1)  # (num_samples, num_bands, seq_len, 2)
target_train_tensor = padded_fluxes_train_tensor  # (num_samples, num_bands, seq_len)
input_test_tensor = torch.stack((padded_times_test_tensor, padded_fluxes_test_tensor), dim=-1)  # (num_samples, num_bands, seq_len, 2)
target_test_tensor = padded_fluxes_test_tensor  # (num_samples, num_bands, seq_len)

# Print shapes for debugging
print(f"Size of concatenated sequence: {sum(max_length_per_band.values())}")
print(f"padded_times_tensor shape: {padded_times_train_tensor.shape}")
print(f"padded_fluxes_tensor shape: {padded_fluxes_train_tensor.shape}")
print(f"input_tensor shape: {input_train_tensor.shape}")
print(f"target_tensor shape: {target_train_tensor.shape}")
print("Examples of the Data:")
print(padded_times_train[0][0],padded_fluxes_train[0][0])
print()
print(f"padded_times_tensor shape: {padded_times_test_tensor.shape}")
print(f"padded_fluxes_tensor shape: {padded_fluxes_test_tensor.shape}")
print(f"input_tensor shape: {input_test_tensor.shape}")
print(f"target_tensor shape: {target_test_tensor.shape}")
print("Examples of the Data:")
print(padded_times_test[0][0],padded_fluxes_test[0][0])
#print(padded_test_idx_all[0])
#print(padded_test_idx_accum[0])
# -

# ### Autoencoder

class RNN_Autoencoder(nn.Module):
    def __init__(self, rnn_input_size, total_input_size, features_size, nn_hidden_size, num_layers=1):
        super(RNN_Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(rnn_input_size, features_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(features_size, nn_hidden_size),  
            nn.ReLU(),
            nn.Linear(nn_hidden_size, total_input_size)  
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_0 = torch.zeros(self.encoder.num_layers, batch_size , self.encoder.hidden_size).to(x.device)
        c_0 = torch.zeros(self.encoder.num_layers, batch_size , self.encoder.hidden_size).to(x.device)
        # Encoder
        _, (hidden, _) = self.encoder(x, (h_0, c_0))  
        hidden = hidden[-1] # Take the hidden state from the last layer only
        #print(hidden.squeeze(0).shape)

        decoded = self.decoder(hidden)
        
        return decoded


autoencoder_rnn_input_size = 2  # Time and flux as input
autoencoder_nn_hidden_size = 1024
autoencoder_total_input_size = sum(max_length_per_band.values())
autoencoder_features_size = 256
autoencoder_num_layers = 2


# ### Build the model

class CustomMultiBandTimeSeriesBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(CustomMultiBandTimeSeriesBRNN, self).__init__()
        # Forward LSTM
        self.lstm_forward = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # Backward LSTM
        self.lstm_backward = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
        # Adding additional neural network layers
        self.intermediate_fc = nn.Linear(hidden_size * 2, hidden_size)  # Additional layer
        self.activation = nn.ReLU()  # Activation layer
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1) 

    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states for forward LSTM
        h_0_forward = torch.zeros(self.lstm_forward.num_layers, batch_size, self.lstm_forward.hidden_size).to(x.device)
        c_0_forward = torch.zeros(self.lstm_forward.num_layers, batch_size, self.lstm_forward.hidden_size).to(x.device)
        # Initialize hidden and cell states for backward LSTM
        h_0_backward = torch.zeros(self.lstm_backward.num_layers, batch_size, self.lstm_backward.hidden_size).to(x.device)
        c_0_backward = torch.zeros(self.lstm_backward.num_layers, batch_size, self.lstm_backward.hidden_size).to(x.device)
        
        # Compute forward and backward outputs
        out_forward, _ = self.lstm_forward(x, (h_0_forward, c_0_forward))
        x_backward = torch.flip(x, [1])
        out_backward, _ = self.lstm_backward(x_backward, (h_0_backward, c_0_backward))
        out_backward = torch.flip(out_backward, [1])

        # Concatenate the outputs from both directions
        out_combined = torch.cat((out_forward, out_backward), dim=1)
        
        # Adjust the indices for output concatenation
        out_forward_adjusted = out_forward[:, :-2, :]  # Remove the last two time step
        out_backward_adjusted = out_backward[:, 2:, :]  # Remove the first two time step

        # Concatenate adjusted outputs
        out_combined = torch.cat((out_forward_adjusted, out_backward_adjusted), dim=2)
        
        # Apply the additional neural network layers
        out_intermediate = self.intermediate_fc(out_combined)
        out_activated = self.activation(out_intermediate)


        # Pass through the fully connected layer
        out = self.fc(out_activated)
        
        return out


class CustomMultiBandTimeSeriesBRNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, autoencoder_rnn_input_size, autoencoder_features_size, autoencoder_num_layers=1):
        super(CustomMultiBandTimeSeriesBRNNAutoencoder, self).__init__()

        # Encoder form Autoencoder
        self.encoder = nn.LSTM(autoencoder_rnn_input_size, autoencoder_features_size, autoencoder_num_layers, batch_first=True)

        # Forward and Backward LSTM layers for BRNN
        self.lstm_forward = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lstm_backward = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)

        # Additional intermediate layer to incorporate encoder outputs
        self.intermediate_fc = nn.Linear(hidden_size * 2 + autoencoder_features_size, hidden_size)  # Adjusted size
        self.activation = nn.ReLU()
        
        # Final output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Use encoder to get encoded features
        _, (hidden, _) = self.encoder(x)  
        encoded_features = hidden[-1] # Take the hidden state from the last layer only
        
        # Forward LSTM processing
        h_0_forward = torch.zeros(self.lstm_forward.num_layers, batch_size, self.lstm_forward.hidden_size).to(x.device)
        c_0_forward = torch.zeros(self.lstm_forward.num_layers, batch_size, self.lstm_forward.hidden_size).to(x.device)
        out_forward, _ = self.lstm_forward(x, (h_0_forward, c_0_forward))

        # Backward LSTM processing
        x_backward = torch.flip(x, [1])
        h_0_backward = torch.zeros(self.lstm_backward.num_layers, batch_size, self.lstm_backward.hidden_size).to(x.device)
        c_0_backward = torch.zeros(self.lstm_backward.num_layers, batch_size, self.lstm_backward.hidden_size).to(x.device)
        out_backward, _ = self.lstm_backward(x_backward, (h_0_backward, c_0_backward))
        out_backward = torch.flip(out_backward, [1])

        # Combining Forward and Backward outputs
        out_forward_adjusted = out_forward[:, :-2, :]  # Remove the last two time step
        out_backward_adjusted = out_backward[:, 2:, :]  # Remove the first two time step
        out_combined = torch.cat((out_forward_adjusted, out_backward_adjusted, encoded_features.unsqueeze(1).expand(-1, seq_len-2, -1)), dim=2)

        # Apply intermediate fully connected layer
        out_intermediate = self.intermediate_fc(out_combined)
        out_activated = self.activation(out_intermediate)

        # Final output layer
        out = self.fc(out_activated)

        return out


# +
# Define the model parameters
input_size = 2  # Time and flux as input
hidden_size = 1024
num_layers = 2

padding_value = -1
# Instantiate the model
#model = CustomMultiBandTimeSeriesBRNNAutoencoder(input_size, hidden_size, num_layers)
model = CustomMultiBandTimeSeriesBRNNAutoencoder(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    autoencoder_rnn_input_size=autoencoder_rnn_input_size,
    autoencoder_features_size=autoencoder_features_size,
    autoencoder_num_layers=autoencoder_num_layers
)


saved_file_path_load = './Autoencoder_runs/Autoencoder_run_2/models/encoder_final_model.pth'
encoder_state_dict = torch.load(saved_file_path_load)
model.encoder.load_state_dict(encoder_state_dict)
# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False


model.to(device)

# Define loss and optimizer
learning_rate = 0.001
criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply mask later
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create a dataset and dataloader
batch_size = 60
dataset = TensorDataset(input_train_tensor, target_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10

# Print the model summary
print(model)
# -

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\nBRNN model structure:\n")
f.write(f"Input Size: {input_size}\n")
f.write(f"Neurons per LSTM layer: {hidden_size}\n")
f.write(f"Number of layers: {num_layers}\n")
f.write(f"Learning Rate: {learning_rate}\n")
f.write(f"Batch Size: {batch_size}\n")
f.write(f"Num Epochs: {num_epochs}\n\n")
print(model,file=f)
f.close()

# ## Train the Model

# ### Custom BRNN

# +
wandb.finish()
project_name = 'BRNN'
run = wandb.init(project=project_name)

# Capture the run name
run_name = run.name  # Get the unique identifier for the current run

# Initialize variables for best model saving
best_loss = float('inf')
best_model_path = os.path.join(base_directory,"models/best_model_checkpoint.pth")

if not os.path.exists(os.path.join(base_directory, 'models')):
    os.makedirs(os.path.join(base_directory, 'models'))

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S %Z")
current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
f.write(f"\nTime: {current_time}\nDate: {current_date}\n")
f.write("\nModel Training\n")
f.write(f"\nProject Name: {run_name}\n")
f.write(f"Run Name: {project_name}\n\n")
total_time = 0


f.close() # Close file
f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')

# Training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.watch(model, log="all")

for epoch in range(num_epochs):
    start_time = time.time()  # Start chrono
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing"):
        # Move data to the device
        inputs, targets = inputs.to(device), targets.to(device)
        
        targets_adjusted = targets[:, 1:-1]
        
        
        # Forward pass
        outputs = model(inputs)
        
        if outputs.shape[1] != targets_adjusted.shape[1]:
            raise ValueError("Output and target sequence lengths do not match.")
        
        # Mask the padded values
        mask = inputs[:, 1:-1, 0] != padding_value
        # Apply the mask to the outputs and targets
        mask_expanded = mask.unsqueeze(-1).expand_as(outputs)
        outputs_masked = outputs[mask_expanded].view(-1)
        targets_masked = targets_adjusted[mask].view(-1)
        
        loss = criterion(outputs_masked, targets_masked).mean()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Log the average loss for the epoch
    avg_loss = running_loss / len(dataloader)
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:f}')
    
    # Save the best model checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        wandb.save("best_model_checkpoint.pth")

    end_time = time.time()  # Stop chrono
    epoch_time = end_time - start_time  
    total_time += epoch_time
    f.write(f"Epoch {epoch + 1} Time: {epoch_time}  Loss: {avg_loss}\n")
    f.close() # Close file
    f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')


f.write(f"\nTotal time: {total_time}\nAvarage Time per Epoch: {total_time/num_epochs}\n")
current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S %Z")
current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
f.write(f"\nTime: {current_time}\nDate: {current_date}\n")
f.close() # Close file

# Save the final model checkpoint
final_model_path = os.path.join(base_directory,"models/final_model_checkpoint.pth")
torch.save(model.state_dict(), final_model_path)
wandb.save("final_model_checkpoint.pth")

print('Training complete.')
# -

# ## Test

# ### Test Predicting all data from 1 AGN

valid_data_found = False
obj_id = 1
while obj_id < len(padded_times_train) and not valid_data_found:
    try:
        # Extract the data for the selected object
        df_lc_single = df_lc.loc[[obj_id]]
        redshifts_single = {obj_id: redshifts[obj_id]}
        
        # Use the existing function to prepare the data for the selected object
        padded_times_single, padded_fluxes_single = unify_lc_for_rnn_multi_band(df_lc_single, redshifts_single, max_length_per_band, bands_inlc=bands_inlc)
        
        # Convert the selected object's data to PyTorch tensors
        padded_times_tensor_single = torch.tensor(padded_times_single, dtype=torch.float32)
        padded_fluxes_tensor_single = torch.tensor(padded_fluxes_single, dtype=torch.float32)
        
        # Combine time and flux into a single input tensor
        input_tensor_single = torch.stack((padded_times_tensor_single, padded_fluxes_tensor_single), dim=-1)  # (num_bands, seq_len, 2)
        
        input_tensor_single = input_tensor_single.to(device)
        
        # Use the model to make predictions
        model.eval()
        with torch.no_grad():
            predictions_single = model(input_tensor_single)
            predictions_single = predictions_single.cpu()  # Move tensor to CPU
        
        print(len(input_tensor_single[0]),len(predictions_single[0]))
        
        valid_data_found = True
    except Exception as e:
        obj_id += 1
        print(f"Error encountered: {e}. Retrying with a different object.")


# +

def process_band_data(padded_times, padded_fluxes, predictions, max_length_per_band, padding_value = -1):
    times_dict = {}
    times_dict_pred = {}
    fluxes_dict = {}
    predictions_dict = {}
    mask_dict = {}
    mask_pred_dict = {}

    start_idx = 0
    start_idx_pred = 0

    for i, (band, length) in enumerate(max_length_per_band.items()):
        end_idx = start_idx + length
        end_idx_pred = start_idx_pred + length

        # Adjusting prediction end index for the first and last bands
        if i == 0 or i == len(max_length_per_band) - 1:
            end_idx_pred -= 1

        # Extracting times, fluxes, and predictions for the current band
        times_band = padded_times[start_idx:end_idx]
        fluxes_band = padded_fluxes[start_idx:end_idx]
        predictions_band = predictions[start_idx_pred:end_idx_pred]

        # Create masks to filter out padding values
        mask = times_band != padding_value
        if i == 0:
            mask_pred = times_band[1:] != padding_value
            times_dict_pred[band] = times_band[1:][mask_pred]
        elif  i == len(max_length_per_band) - 1:
            mask_pred = times_band[:-1] != padding_value
            times_dict_pred[band] = times_band[:-1][mask_pred]
        else: 
            mask_pred = mask
            times_dict_pred[band] = times_band[mask_pred]
            
        # Applying masks
        times_dict[band] = times_band[mask]
        fluxes_dict[band] = fluxes_band[mask]
        predictions_dict[band] = predictions_band[mask_pred]
        mask_dict[band] = mask
        mask_pred_dict[band] = mask_pred

        # Update indices for the next band
        start_idx = end_idx
        start_idx_pred = end_idx_pred

    return times_dict, fluxes_dict, times_dict_pred, predictions_dict, mask_dict, mask_pred_dict






def plot_band_data(padded_times, padded_fluxes, predictions, max_length_per_band, padding_value=-1):
    # Start index for each band in the concatenated array
    fig, axes = plt.subplots(nrows=len(bands_inlc), figsize=(14, 10))
    # Adjust first and last item not predicted
    padded_times_adj = padded_times[0]
    padded_fluxes_adj = padded_fluxes[0]


    times, fluxes, times_pred, predictions, masks, mask_preds = process_band_data(padded_times[0], padded_fluxes[0], predictions[0], max_length_per_band, padding_value)
    # Iterate over each band using the lengths specifiedprint
    for i, (band, length) in enumerate(max_length_per_band.items()):
        
        # Inverse normalization
        predictions_band_scaled = flux_scaler.inverse_transform(predictions[band].reshape(-1, 1))
        #predictions_band_scaled = predictions_band_scaled.reshape(predictions_band.shape)
        fluxes_band_scaled = flux_scaler.inverse_transform(fluxes[band].reshape(-1, 1))
        times_band_scaled = time_scaler.inverse_transform(times[band].reshape(-1, 1))
        times_pred_band_scaled = time_scaler.inverse_transform(times_pred[band].reshape(-1, 1))
        
        ax = axes[i]
        # Plot training and testing data
        ax.plot(times_band_scaled, fluxes_band_scaled, '-o', color=colors[i], label='Actual Flux', alpha=0.7)
        ax.plot(times_pred_band_scaled, predictions_band_scaled, '-x', color='black', label='Predicted Flux', markersize=4, markeredgewidth=2)
        # Setting plot titles and labels
        ax.set_title(f'Flux Prediction for Band {band} - OID {obj_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})

    fig_title = 'Prediction of All AGN Data'
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    if not os.path.exists(os.path.join(base_directory,'output')):
        os.makedirs(os.path.join(base_directory,'output'))
    save_path = os.path.join(os.path.join(base_directory,'output'), fig_title.replace(' ', '_') + '.png')
    fig.savefig(save_path)
    print(f"Saved figure to {save_path}")
        
    
plot_band_data(padded_times_tensor_single.numpy(), padded_fluxes_tensor_single.numpy(), predictions_single, max_length_per_band)
# -

# ### Test AGN with train set

valid_data_found = False
obj_id = 0
while obj_id < len(padded_times_train) and not valid_data_found:
    try:
        
        # Convert the selected object's data to PyTorch tensors
        padded_times_tensor_single = torch.tensor([padded_times_train[obj_id]], dtype=torch.float32)
        padded_fluxes_tensor_single = torch.tensor([padded_fluxes_train[obj_id]], dtype=torch.float32)
        
        # Combine time and flux into a single input tensor
        input_tensor_single = torch.stack((padded_times_tensor_single, padded_fluxes_tensor_single), dim=-1)  # (num_bands, seq_len, 2)
        target_tensor_single = padded_fluxes_tensor_single.unsqueeze(-1)  # (num_bands, seq_len, 1)
        
        input_tensor_single = input_tensor_single.to(device)
        
        # Use the model to make predictions
        model.eval()
        with torch.no_grad():
            predictions_single = model(input_tensor_single)
            predictions_single = predictions_single.cpu()  # Move tensor to CPU
        
        valid_data_found = True
    except Exception as e:
        obj_id += 1
        print(f"Error encountered: {e}. Retrying with a different object.")


# +
def plot_band_data_train(padded_times, padded_fluxes, predictions, max_length_per_band, padding_value=-1):
    # Start index for each band in the concatenated array
    fig, axes = plt.subplots(nrows=len(bands_inlc), figsize=(14, 10))
    # Adjust first and last item not predicted
    padded_times_adj = padded_times[0]
    padded_fluxes_adj = padded_fluxes[0]


    times, fluxes, times_pred, predictions, masks, mask_preds = process_band_data(padded_times[0], padded_fluxes[0], predictions[0], max_length_per_band, padding_value)
    # Iterate over each band using the lengths specifiedprint
    for i, (band, length) in enumerate(max_length_per_band.items()):
        
        # Inverse normalization
        predictions_band_scaled = flux_scaler.inverse_transform(predictions[band].reshape(-1, 1))
        fluxes_band_scaled = flux_scaler.inverse_transform(fluxes[band].reshape(-1, 1))
        times_band_scaled = time_scaler.inverse_transform(times[band].reshape(-1, 1))
        times_pred_band_scaled = time_scaler.inverse_transform(times_pred[band].reshape(-1, 1))
        
        ax = axes[i]
        # Plot training and testing data
        ax.plot(times_band_scaled, fluxes_band_scaled, '-o', color=colors[i], label='Actual Flux', alpha=0.7)
        ax.plot(times_pred_band_scaled, predictions_band_scaled, '-x', color='black', label='Predicted Flux', markersize=4, markeredgewidth=2)
        # Setting plot titles and labels
        ax.set_title(f'Flux Prediction for Band {band} - OID {obj_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})

    fig_title = 'Prediction of train AGN Data'
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    if not os.path.exists(os.path.join(base_directory,'output')):
        os.makedirs(os.path.join(base_directory,'output'))
    save_path = os.path.join(os.path.join(base_directory,'output'), fig_title.replace(' ', '_') + '.png')
    fig.savefig(save_path)
    print(f"Saved figure to {save_path}")
        

plot_band_data_train(padded_times_tensor_single.numpy(), padded_fluxes_tensor_single.numpy(), predictions_single, max_length_per_band_train)
# -

# ### Test AGN with test set

valid_data_found = False
obj_id = 0
while obj_id < len(padded_times_train) and not valid_data_found:
    try:
        
        # Convert the selected object's data to PyTorch tensors
        padded_times_tensor_single = torch.tensor([padded_times_test[obj_id]], dtype=torch.float32)
        padded_fluxes_tensor_single = torch.tensor([padded_fluxes_test[obj_id]], dtype=torch.float32)
        
        # Combine time and flux into a single input tensor
        input_tensor_single = torch.stack((padded_times_tensor_single, padded_fluxes_tensor_single), dim=-1)  # (num_bands, seq_len, 2)
        target_tensor_single = padded_fluxes_tensor_single.unsqueeze(-1)  # (num_bands, seq_len, 1)
        
        input_tensor_single = input_tensor_single.to(device)
        
        # Use the model to make predictions
        model.eval()
        with torch.no_grad():
            predictions_single = model(input_tensor_single)
            predictions_single = predictions_single.cpu()  # Move tensor to CPU
        
        
        valid_data_found = True
    except Exception as e:
        obj_id += 1
        print(f"Error encountered: {e}. Retrying with a different object.")


# +
def plot_band_data_test(padded_times_train, padded_fluxes_train, padded_times_test, padded_fluxes_test, padded_test_idx_all, predictions, max_length_per_band_train, max_length_per_band_test,max_length_per_band, padding_value=-1):
    # Start index for each band in the concatenated array
    start_idx_train = 0
    start_idx_test = 0
    fig, axes = plt.subplots(nrows=len(bands_inlc), figsize=(14, 10))

    times_test, fluxes_test, times_test_pred, predictions, masks, mask_preds = process_band_data(padded_times_test, padded_fluxes_test, predictions[0], max_length_per_band, padding_value)
    
    for i, band in enumerate(bands_inlc):
        length_train = max_length_per_band_train[band]
        # End index for the current band
        end_idx_train = start_idx_train + length_train
        
        # Extract times, fluxes, and predictions for the current band
        times_band_train = padded_times_train[start_idx_train:end_idx_train]
        fluxes_band_train = padded_fluxes_train[start_idx_train:end_idx_train]
        
        # Create a mask to filter out the padding values
        mask_train = times_band_train != padding_value
        times_band_train = times_band_train[mask_train]
        fluxes_band_train = fluxes_band_train[mask_train]
        # Inverse normalization
        predictions_band_scaled = flux_scaler.inverse_transform(predictions[band].reshape(-1, 1))
        fluxes_band_scaled_train = flux_scaler.inverse_transform(fluxes_band_train.reshape(-1, 1))
        times_band_scaled_train = time_scaler.inverse_transform(times_band_train.reshape(-1, 1))
        fluxes_band_scaled_test = flux_scaler.inverse_transform(fluxes_test[band].reshape(-1, 1))
        times_band_scaled_test = time_scaler.inverse_transform(times_test[band].reshape(-1, 1))
        times_pred_band_scaled = time_scaler.inverse_transform(times_test_pred[band].reshape(-1, 1))
        # Move the start index to the next band
        start_idx_train = end_idx_train
        
        ax = axes[i]
        # Plot training and testing data
        ax.plot(times_band_scaled_train, fluxes_band_scaled_train, '-o', color=colors[i], label='Actual Flux', alpha=0.7)

        # Plot predictions (assuming predictions are aligned with times after n_steps)
        ax.plot(times_band_scaled_test[padded_test_idx_all[band]], fluxes_band_scaled_test[padded_test_idx_all[band]], '*', color='r', label='Actual Flux', markersize=6, alpha=0.9)


        
        if i == 0:
            adjusted_indexes = padded_test_idx_all[band] - 1
            ax.plot(times_band_scaled_test[padded_test_idx_all[band]], predictions_band_scaled[adjusted_indexes], 'x', color='black', label='Predicted Test Flux', markersize=4, markeredgewidth=2)
        elif  i == len(max_length_per_band) - 1: # Same indexe as only eliminated the last one
            ax.plot(times_band_scaled_test[padded_test_idx_all[band]], predictions_band_scaled[padded_test_idx_all[band]], 'x', color='black', label='Predicted Test Flux', markersize=4, markeredgewidth=2)
        else: 
            ax.plot(times_band_scaled_test[padded_test_idx_all[band]], predictions_band_scaled[padded_test_idx_all[band]], 'x', color='black', label='Predicted Test Flux', markersize=4, markeredgewidth=2)
        
        # Setting plot titles and labels
        ax.set_title(f'Flux Prediction for Band {band} - OID {obj_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    
    fig_title = 'Test Set Prediction'
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    if not os.path.exists(os.path.join(base_directory,'output')):
        os.makedirs(os.path.join(base_directory,'output'))
    save_path = os.path.join(os.path.join(base_directory,'output'), fig_title.replace(' ', '_') + '.png')
    fig.savefig(save_path)
    print(f"Saved figure to {save_path}")


plot_band_data_test(padded_times_train[obj_id],padded_fluxes_train[obj_id],padded_times_test[obj_id],padded_fluxes_test[obj_id],padded_test_idx_all[obj_id], predictions_single, max_length_per_band_train, max_length_per_band_test,max_length_per_band)
# -

# ### Evaluate the Model

# +
# Setup DataLoader for the test data
test_dataset = TensorDataset(input_test_tensor, target_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Typically we do not shuffle test data

# Function to evaluate the model on test data

def evaluate_model(model, dataloader, padded_test_idx_accum, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    count = 0
    
    with torch.no_grad():  # No need to track gradients during evaluation
        index_acc = 0  # Accumulator for the index in padded_test_idx_accum
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            # Add missing values to the outputs filling with targets
            outputs = torch.cat((targets[:, 0:1], outputs.squeeze(-1), targets[:, -2:-1]), dim=1)
            
            if outputs.shape[1] != targets.shape[1]:
                raise ValueError("Output and target sequence lengths do not match.")
            
            outputs_masked = []
            targets_masked = []
            for batch_idx in range(len(outputs)):
                current_indices = padded_test_idx_accum[index_acc]
                outputs_masked.extend(outputs[batch_idx][current_indices])
                targets_masked.extend(targets[batch_idx][current_indices])
                index_acc += 1
                
            outputs_masked = torch.stack(outputs_masked, dim=0).squeeze()
            targets_masked = torch.stack(targets_masked, dim=0).squeeze()
            
            loss = criterion(outputs_masked, targets_masked).mean()
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
    
    return total_loss / count  # Return the average loss

# Call the function to evaluate the model
avg_test_loss = evaluate_model(model, test_dataloader,padded_test_idx_accum, criterion, device)
print(f'Average Test Loss: {avg_test_loss:f}')

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\n\nTest Loss: {avg_test_loss}\n")
f.close()


#log the test loss to wandb
wandb.log({"test_loss": avg_test_loss})

# Finish the wandb session
wandb.finish()

print("Test evaluation complete.")
# -

# ## Save the Model

if not os.path.exists(os.path.join(base_directory, 'models')):
    os.makedirs(os.path.join(base_directory, 'models'))
torch.save(model.state_dict(), os.path.join(base_directory, 'models/final_model_backup.pth'))
torch.save(model.state_dict(), os.path.join(base_directory, 'models/final_model.pth'))


def save_model(model, directory, base_filename):
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Initialize filename
    filename = f"{base_filename}_1.pth"
    file_path = os.path.join(directory, filename)
    
    # Check if the file already exists and increment the number in the filename until it doesn't
    counter = 1
    while os.path.exists(file_path):
        counter += 1
        filename = f"{base_filename}_{counter}.pth"
        file_path = os.path.join(directory, filename)
    
    # Save the model
    torch.save(model.state_dict(), file_path)
    print(f"Model saved as {file_path}")
    return file_path


# +
saved_file_path = save_model(model, './models/BRNN_1', 'final_model')

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"Model saved in: {saved_file_path}\n")
f.write(f"Model saved in: {os.path.join(base_directory, 'models/final_model.pth')}\n")
f.close()
# -

# ## Log End

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S %Z")
current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
f.write(f"\nRun Completed\nTime: {current_time}\nDate: {current_date}\n")
f.close()

# ## Load the Model

saved_file_path_load = './BRRN_runs/BRRN_run_11/models/best_model_checkpoint.pth'

# +
#model_load= CustomMultiBandTimeSeriesBRNN(input_size, hidden_size, num_layers)
#model_load.load_state_dict(torch.load(saved_file_path_load))
#model_load.to(device)  # Ensure you move the model to the appropriate device

# +
#model = model_load
