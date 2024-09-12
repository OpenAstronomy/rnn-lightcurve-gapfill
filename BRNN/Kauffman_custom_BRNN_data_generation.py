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
subset_size = 100
obj_ids_subset = df_lc.index.get_level_values('objectid').unique()[:subset_size]
#obj_ids_subset = df_lc.index.get_level_values('objectid').unique()

# Extract the data for the selected objects
df_lc_subset = df_lc.loc[obj_ids_subset]
redshifts_subset = {obj_id: redshifts[obj_id] for obj_id in obj_ids_subset}


# -

# #### Fill the sequences with padding

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
            #print(len(obj_times),sum(max_length_per_band.values())) 
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
                        #print(len(x_train), len(x_test),  max_length_per_band_test[band], padded_x_test)
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
    
    #print(max_length_per_band_test)
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

# ### Custom BRNN

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


# +
# Define the model parameters
input_size = 2  # Time and flux as input
#hidden_size = 2048
hidden_size = 1024
#hidden_size = 512
#hidden_size = 256
#hidden_size = 64
#hidden_size = 32
num_layers = 2

padding_value = -1
# Instantiate the model
model = CustomMultiBandTimeSeriesBRNN(input_size, hidden_size, num_layers)
model.to(device)

# Define loss and optimizer
learning_rate = 0.001
criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply mask later
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create a dataset and dataloader
#batch_size = 64
batch_size = 32
dataset = TensorDataset(input_train_tensor, target_train_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10

# Print the model summary
print(model)
# -

# ## Load the Model

# ##### 1024 final

#saved_file_path_load = './BRRN_runs/BRRN_run_20_512/models/best_model_checkpoint.pth'
saved_file_path_load = './BRRN_runs/BRRN_run_21_1024/models/final_model.pth'

model_load= CustomMultiBandTimeSeriesBRNN(input_size, hidden_size, num_layers)
model_load.load_state_dict(torch.load(saved_file_path_load))
model_load.to(device) 

model = model_load


# ## Test

# ### Predict Missing values

def find_missing_times(agn_id, dataframe, bands, max_time_coverage, time_scaler):
    """
    Identifies missing time points for a specified AGN across given bands.

    Args:
    - agn_id (int): The object ID of the AGN for which to find missing times.
    - dataframe (pd.DataFrame): The DataFrame containing time series data for AGNs.
    - bands (list): A list of bands to check for missing time points.
    - max_time_coverage (dict): A dictionary with bands as keys and arrays of times representing maximum coverage as values.

    Returns:
    - dict: A dictionary with bands as keys and arrays of missing times as values.
    """
    missing_time_coverage = {}

    for band in bands:
        # Get all times for the specified AGN in the current band
        agn_times = dataframe.xs((agn_id, band), level=('objectid', 'band')).index.get_level_values('time').unique()

        # Get the times for the AGN with the most measurements in the current band
        max_coverage_times = max_time_coverage[band]

        # Determine missing times by finding set differences
        missing_times = np.setdiff1d(max_coverage_times, agn_times)
        #missing_time_coverage[band] = missing_times
        
        missing_time_coverage[band] = time_scaler.transform(missing_times.reshape(-1, 1)).flatten()

    return missing_time_coverage


# +
first_oid = df_lc.index.get_level_values('objectid').unique()[0]  # Just for demonstration, replace with any specific AGN ID

# Get missing times for the first AGN
missing_time_coverage_per_band = find_missing_times(first_oid, df_lc, bands_inlc, max_time_coverage_per_band,time_scaler)


# +
def input_sequence_missing_data(obj_id, missing_time_coverage, df_lc, max_length_per_band, bands_inlc, padding_value=-1):
    #objids = df_lc.index.get_level_values('objectid').unique()
    #if isinstance(redshifts, np.ndarray):
    #    redshifts = dict(zip(objids, redshifts))
        
    obj_times, obj_fluxes= [], []
    missing_indexes_band = {}
    band_lenghts = {}
    #redshift = redshifts.get(obj_id, None)
    singleobj = df_lc.loc[obj_id]
    label = singleobj.index.unique('label')[0]
    bands = singleobj.index.get_level_values('band').unique()

    if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
        cumulative_length = 0  # This tracks the length of the sequence before adding the current band
        for band in bands_inlc:
            if (label, band) in singleobj.index:
                band_lc = singleobj.xs((label, band), level=('label', 'band'))
                band_lc_clean = band_lc[(band_lc.index.get_level_values('time') > 56000) & (band_lc.index.get_level_values('time') < 65000)]
                x = np.array(band_lc_clean.time_norm)
                y = np.array(band_lc_clean.flux_norm)

                #sorted_indices = np.argsort(x)
                #x = x[sorted_indices]
                #y = y[sorted_indices]
                
                
                    
                # Find missing times for the current band and calculate the mean of the existing fluxes
                missing_times = missing_time_coverage[band] if band in missing_time_coverage else []
                mean_flux = np.mean(y) if y.size > 0 else 0
                
                # Include missing times in the x array and mean flux value in the y array
                complete_x = np.concatenate([x, missing_times])
                missing_fluxes = np.full_like(missing_times, fill_value=mean_flux)
                complete_y = np.concatenate([y, missing_fluxes])

                # Resort the arrays by time
                resort_indices = np.argsort(complete_x)
                complete_x = complete_x[resort_indices]
                complete_y = complete_y[resort_indices]
        
                
                # Determine the indexes of the missing times in the sorted array
                inserted_indexes = np.searchsorted(complete_x, missing_times)
                #inserted_indexes = inserted_indexes + cumulative_length
                #cumulative_length += len(complete_x) 
                #missing_indexes.extend(inserted_indexes.tolist())
                missing_indexes_band[band]= inserted_indexes.tolist()
                
                obj_times.extend(complete_x)
                obj_fluxes.extend(complete_y)
                band_lenghts[band] = len(complete_x)
                    
    return obj_times, obj_fluxes, band_lenghts, missing_indexes_band

seq_times, seq_fluxes, band_lenghts, missing_indexes_band = input_sequence_missing_data(first_oid,missing_time_coverage_per_band,df_lc_subset, max_length_per_band,bands_inlc)
seq_times = [seq_times]
seq_fluxes = [seq_fluxes]
# Convert the data to PyTorch tensors
times_tensor = torch.tensor(seq_times, dtype=torch.float32)
fluxes_tensor = torch.tensor(seq_fluxes, dtype=torch.float32)

# Combine time and flux into a single input tensor
input_tensor = torch.stack((times_tensor, fluxes_tensor), dim=-1)  # (num_samples, num_bands, seq_len, 2)

# Print shapes for debugging

print(f"padded_times_tensor shape: {times_tensor.shape}")
print(f"padded_fluxes_tensor shape: {fluxes_tensor.shape}")
print(f"input_tensor shape: {input_tensor.shape}")


# +
def interpolate_model(model, input_tensor):    
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        predictions_from_model = model(input_tensor)
        predictions_from_model = predictions_from_model.cpu()
    
        first_flux = input_tensor[:, 0, 1]
        last_flux = input_tensor[:, -1, 1]
        first_flux_tensor = first_flux.unsqueeze(1).unsqueeze(2).expand(-1, 1, predictions_from_model.shape[2]).cpu()
        last_flux_tensor = last_flux.unsqueeze(1).unsqueeze(2).expand(-1, 1, predictions_from_model.shape[2]).cpu()
        
        predictions_single = torch.cat((first_flux_tensor, predictions_from_model, last_flux_tensor), dim=1)
        predictions_single = predictions_single.cpu()  # Move tensor to CPU
    return predictions_single
predictions_single = interpolate_model(model, input_tensor)

print(f"padded_fluxes_tensor shape: {predictions_single.shape}")


# +
def plot_band_data_without_mask(times, fluxes, predictions, band_lenghts, missing_indexes_band, padding_value=-1):
    # Start index for each band in the concatenated array
    start_idx = 0
    fig, axes = plt.subplots(nrows=len(bands_inlc), figsize=(14, 10))
    colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for each band
    
    # Iterate over each band using the lengths specified
    for i, (band, length) in enumerate(band_lenghts.items()):
        # End index for the current band
        end_idx = start_idx + length
        
        # Extract times, fluxes, and predictions for the current band
        times_band = times[0][start_idx:end_idx]
        fluxes_band = fluxes[0][start_idx:end_idx]
        predictions_band = predictions[0][start_idx:end_idx]
        
        
        # Inverse normalization
        predictions_band_scaled = flux_scaler.inverse_transform(predictions_band.reshape(-1, 1))
        #predictions_band_scaled = predictions_band_scaled.reshape(predictions_band.shape)
        fluxes_band_scaled = flux_scaler.inverse_transform(fluxes_band.reshape(-1, 1))
        times_band_scaled = time_scaler.inverse_transform(times_band.reshape(-1, 1))
        # Move the start index to the next band
        start_idx = end_idx
        
        ax = axes[i]
        
        # Plot data
        
        combined_fluxes = [predictions_band_scaled[i] if i in missing_indexes_band[band] else flux for i, flux in enumerate(fluxes_band_scaled)]
        ax.plot(times_band_scaled, combined_fluxes , '-*', color='r', label='Actual Flux', alpha=0.7)
        
        ax.plot([times_band_scaled[i] for i in range(len(times_band_scaled)) if i not in missing_indexes_band[band]], [fluxes_band_scaled[i] for i in range(len(times_band_scaled)) if i not in missing_indexes_band[band]] , '-o', color=colors[i], label='Actual Flux', alpha=0.7)
       
        # Plot predictions (assuming predictions are aligned with times after n_steps)
        ax.plot([times_band_scaled[i] for i in missing_indexes_band[band]], [predictions_band_scaled[i] for i in missing_indexes_band[band]], 'x', color='black', label='Predicted Test Flux', markersize=4, markeredgewidth=2)

        # Setting plot titles and labels
        ax.set_title(f'Flux Prediction for Band {band} ')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.legend()
        #print(len(times_band))

    plt.tight_layout()
    plt.show()
    
plot_band_data_without_mask(times_tensor.numpy(), fluxes_tensor.numpy(), predictions_single, band_lenghts, missing_indexes_band)
# -

# #### All AGN

# ###### Load missing times dictionary

import pickle
with open('data/generated/missing_present_times_dict.pkl', 'rb') as handle:
    missing_times_dict = pickle.load(handle)


def input_sequence_missing_data(obj_id, missing_time_coverage, df_lc, max_length_per_band, bands_inlc, padding_value=-1):
        
    obj_times, obj_fluxes= [], []
    missing_indexes_band = {}
    band_lenghts = {}
    singleobj = df_lc.loc[obj_id]
    label = singleobj.index.unique('label')[0]
    bands = singleobj.index.get_level_values('band').unique()

    if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
        cumulative_length = 0  # This tracks the length of the sequence before adding the current band
        for band in bands_inlc:
            if (label, band) in singleobj.index:
                band_lc = singleobj.xs((label, band), level=('label', 'band'))
                band_lc_clean = band_lc[(band_lc.index.get_level_values('time') > 56000) & (band_lc.index.get_level_values('time') < 65000)]
                x = np.array(band_lc_clean.time_norm)
                y = np.array(band_lc_clean.flux_norm)
                
                
                    
                # Find missing times for the current band and calculate the mean of the existing fluxes
                missing_times = missing_time_coverage[band] if band in missing_time_coverage else []
                mean_flux = np.mean(y) if y.size > 0 else 0
                
                # Include missing times in the x array and mean flux value in the y array
                complete_x = np.concatenate([x, missing_times])
                missing_fluxes = np.full_like(missing_times, fill_value=mean_flux)
                complete_y = np.concatenate([y, missing_fluxes])

                # Resort the arrays by time
                resort_indices = np.argsort(complete_x)
                complete_x = complete_x[resort_indices]
                complete_y = complete_y[resort_indices]
        
                
                # Determine the indexes of the missing times in the sorted array
                inserted_indexes = np.searchsorted(complete_x, missing_times)
                missing_indexes_band[band]= inserted_indexes.tolist()
                
                obj_times.extend(complete_x)
                obj_fluxes.extend(complete_y)
                band_lenghts[band] = len(complete_x)
                    
    return obj_times, obj_fluxes, band_lenghts, missing_indexes_band


def interpolate_model(model, input_tensor):    
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        predictions_from_model = model(input_tensor)
        predictions_from_model = predictions_from_model.cpu()
    
        first_flux = input_tensor[:, 0, 1]
        last_flux = input_tensor[:, -1, 1]
        first_flux_tensor = first_flux.unsqueeze(1).unsqueeze(2).expand(-1, 1, predictions_from_model.shape[2]).cpu()
        last_flux_tensor = last_flux.unsqueeze(1).unsqueeze(2).expand(-1, 1, predictions_from_model.shape[2]).cpu()
        
        predictions_single = torch.cat((first_flux_tensor, predictions_from_model, last_flux_tensor), dim=1)
        predictions_single = predictions_single.cpu()  # Move tensor to CPU
    return predictions_single


# +
from collections import defaultdict

complete_flux_data  = defaultdict(list)



for agn_id in tqdm(df_lc_subset.index.get_level_values('objectid').unique(), desc="Processing AGNs"):

    
    missing_time_coverage_per_band = {}
    for band in bands_inlc:
        #print(missing_times_dict[agn_id][band]['missing_times'])
        if(len(missing_times_dict[agn_id][band]['missing_times']) > 0):
            if isinstance(missing_times_dict[agn_id][band]['missing_times'], pd.Index):
                missing_time_coverage_per_band[band] = time_scaler.transform(missing_times_dict[agn_id][band]['missing_times'].to_numpy().reshape(-1, 1)).flatten()
            else:
                missing_time_coverage_per_band[band] = time_scaler.transform(missing_times_dict[agn_id][band]['missing_times'].reshape(-1, 1)).flatten()
        else:
            missing_time_coverage_per_band[band] = []
    #print(missing_time_coverage_per_band)
    times, fluxes, band_lenghts, missing_indexes_band = input_sequence_missing_data(agn_id,missing_time_coverage_per_band,df_lc_subset, max_length_per_band,bands_inlc)
    times = [times]
    fluxes = [fluxes]
    times_tensor = torch.tensor(times, dtype=torch.float32)
    fluxes_tensor = torch.tensor(fluxes, dtype=torch.float32)
    input_tensor = torch.stack((times_tensor, fluxes_tensor), dim=-1)  # (num_samples, num_bands, seq_len, 2)
    if len(band_lenghts) > 0:
        predictions = interpolate_model(model, input_tensor)
        start_idx = 0
        for i, (band, length) in enumerate(band_lenghts.items()):
                band_data = missing_times_dict[agn_id][band]
                missing_times = band_data['missing_times']
                present_times = band_data['present_times']
                present_fluxes = band_data['present_fluxes']
            
                # End index for the current band
                end_idx = start_idx + length
                #print(length,missing_indexes_band[band])
                # Extract times, fluxes, and predictions for the current band
                times_band = times[0][start_idx:end_idx]
                #print(times_band)
                fluxes_band = fluxes[0][start_idx:end_idx]
                predictions_band = predictions[0][start_idx:end_idx]
                missing_predictions_band = [predictions_band[i] for i in missing_indexes_band[band]]
                missing_predictions_band = np.array([t.item() for t in missing_predictions_band])
                missing_times_band = [times_band[i] for i in missing_indexes_band[band]] 
                missing_times_band = np.array([t.item() for t in missing_times_band])
                #print(missing_predictions_band)
                # Inverse normalization
                if(len(missing_times_dict[agn_id][band]['missing_times']) > 0):
                    missing_predictions_band_scaled = flux_scaler.inverse_transform(missing_predictions_band.reshape(-1, 1)).flatten()
                    missing_times_band_scaled = time_scaler.inverse_transform(missing_times_band.reshape(-1, 1)).flatten()
                    
                # Move the start index to the next band
                start_idx = end_idx
            
                nan_count = np.isnan(missing_predictions_band_scaled).sum()
                if nan_count > 0:
                    print(f"ADGN_ID: {agn_id}, band: {band}. Se encontraron {nan_count} valores NaN después de la interpolación.")
                #print(missing_predictions_band_scaled,present_times, missing_times_band_scaled)
                combined_times = np.concatenate([present_times, missing_times_band_scaled])
                combined_fluxes = np.concatenate([present_fluxes, missing_predictions_band_scaled])
                
                sorted_indices = np.argsort(combined_times)
                sorted_times = combined_times[sorted_indices]
                sorted_fluxes = combined_fluxes[sorted_indices]
                
                complete_flux_data[agn_id].extend(sorted_fluxes)
    
# Convert the dictionary to a DataFrame for saving
final_data = pd.DataFrame([(key, np.array(value)) for key, value in complete_flux_data.items()], columns=['objectid', 'flux_array'])


# +

print(len(complete_flux_data[agn_id]))
print(len(complete_flux_data[5]))
# -

del complete_flux_data[agn_id]

print(missing_times_dict[agn_id][band], agn_id)

print(len(missing_times_dict[1][band]['missing_times'])+len(missing_times_dict[1][band]['present_times']))

# ##### Save Dataset

final_data.to_parquet('data/generated/model_data.parquet')

with open('data/generated/model_data.pkl', 'wb') as f:
    pickle.dump(complete_flux_data, f)

print(final_data)


final_data_1000 = final_data

print(len(final_data.iloc[70]['flux_array']))

print(missing_times_dict[1])
