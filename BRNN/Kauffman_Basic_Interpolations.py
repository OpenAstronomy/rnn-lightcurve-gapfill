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
from sklearn.metrics import mean_squared_error

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

# ## Log File

# +
counter = 1
base_directory = f"./Basic_runs/Basic_run_{counter}"

# Check existing directories and increment to find the next available 'run_x' directory
while os.path.exists(base_directory):
    counter += 1
    base_directory = f"./Basic_runs/Basic_run_{counter}"

# Create the new run directory
os.makedirs(base_directory)
print(f"New run directory created: {base_directory}")
# -

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S %Z")
current_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
f.write(f"\n------------------------------------------\n\nBRRN run\nTime: {current_time}\nDate: {current_date}\n")
f.close()

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Loading data
# -

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

# +
subset_size = 10000
obj_ids_subset = df_lc.index.get_level_values('objectid').unique()[:subset_size]
#obj_ids_subset = df_lc.index.get_level_values('objectid').unique()

# Extract the data for the selected objects
df_lc_subset = df_lc.loc[obj_ids_subset]
# -

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
    # Extraer datos para cada banda del AGN con más mediciones
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

import pickle
# Save the dictionary to a pickle file
#with open('data/generated/max_time_coverage_per_band_def.pickle', 'wb') as handle:
    #pickle.dump(max_time_coverage_per_band, handle, protocol=pickle.HIGHEST_PROTOCOL)


# +
# Load the dictionary from the pickle file
#with open('data/generated/max_time_coverage_per_band_def.pickle', 'rb') as handle:
    #loaded_max_time_coverage_per_band = pickle.load(handle)

# Now you can access the loaded dictionary like a regular dictionary
#print(loaded_max_time_coverage_per_band['W1'])
# -

# ### Obtain time gaps of AGN 

def find_missing_times(agn_id, dataframe, bands, max_time_coverage):
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
        missing_time_coverage[band] = missing_times

    return missing_time_coverage


# #### First AGN

# +
first_oid = df_lc.index.get_level_values('objectid').unique()[0]  # Just for demonstration, replace with any specific AGN ID

# Get missing times for the first AGN
missing_time_coverage_per_band = find_missing_times(first_oid, df_lc, bands_inlc, max_time_coverage_per_band)

# Print the results
print("Missing Time Coverage for the AGN Per Band:")
for band, times in missing_time_coverage_per_band.items():
    print(f"Band {band}: {times if len(times) > 0 else 'No missing times'}")


# -

# #### All AGN

# ##### With present fluxes

# +
def find_missing_and_present_times(agn_id, singleobj, bands, max_time_coverage):
    time_coverage = {}
    label = singleobj.index.unique('label')[0]

    for band in bands:
        # Inicializa estructuras para guardar los tiempos que faltan y los que están presentes
        time_coverage[band] = {
            'missing_times': [],
            'present_times': [],
            'present_fluxes': []
        }
        
        if (label, band) in singleobj.index:
            # Obtiene datos para la banda actual
            band_data = singleobj.xs((label, band), level=('label', 'band'))
            agn_times = band_data.index.get_level_values('time').unique()
            agn_fluxes = band_data['flux']

            # Tiempos máximos de cobertura para la banda
            max_coverage_times = max_time_coverage[band]

            # Encuentra tiempos que faltan y tiempos presentes
            missing_times = np.setdiff1d(max_coverage_times, agn_times)
            present_times = np.intersect1d(agn_times, max_coverage_times)
            present_fluxes = band_data.loc[band_data.index.get_level_values('time').isin(present_times), 'flux']

            # Guarda los resultados en el diccionario
            time_coverage[band]['missing_times'] = missing_times
            time_coverage[band]['present_times'] = present_times
            time_coverage[band]['present_fluxes'] = present_fluxes.values
        else:
            # Si no hay datos para la banda, se asignan arrays vacíos
            max_coverage_times = max_time_coverage[band]
            time_coverage[band]['missing_times'] = max_coverage_times

    return time_coverage

def find_missing_times_for_all_agns(dataframe, bands, max_time_coverage):
    all_missing_times = {}

    # Iterate over each unique AGN ID
    for agn_id in tqdm(dataframe.index.get_level_values('objectid').unique(), desc="Processing AGNs"):
        singleobj = df_lc.loc[agn_id]
        missing_times = find_missing_and_present_times(agn_id, singleobj, bands, max_time_coverage)
        all_missing_times[agn_id] = missing_times

    return all_missing_times


# -

missing_times_dict = find_missing_times_for_all_agns(df_lc, bands_inlc, max_time_coverage_per_band)
missing_times_df = pd.DataFrame.from_dict(missing_times_dict, orient='index')

agns_with_present_times = [agn_id for agn_id, bands in missing_times_dict.items() if any(len(bands[band]['present_times']) > 0 for band in bands)]
print("AGN IDs with present times:", len(agns_with_present_times))
print(agns_with_present_times)

# +
import pickle

with open('data/generated/missing_present_times_dict_10000.pkl', 'wb') as handle:
    pickle.dump(missing_times_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -

with open('data/generated/missing_present_times_dict_10000.pkl', 'rb') as handle:
    loaded_missing_times_dict = pickle.load(handle)

# ##### Save Data

missing_times_df.to_parquet('data/generated/missing_times_per_agn.parquet')

# #### Load Missing Times

with open('data/generated/missing_present_times_dict.pkl', 'rb') as handle:
    missing_times_dict = pickle.load(handle)

# +
#missing_times_df = pd.read_parquet('data/generated/missing_times_per_agn_10000.parquet')
#print(missing_times_df.loc[0])
# -

# ### Test Gaps

# +
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def split_covered_indices(agn_id, dataframe, bands, test_size=0.2):
    """
    Splits the positional indices of covered times of a specified AGN into training and testing sets for each band,
    ensuring that the indices are ordered.

    Args:
    - agn_id (int): The object ID of the AGN for which to split indices.
    - dataframe (pd.DataFrame): The DataFrame containing time series data for AGNs.
    - bands (list): A list of bands to check for time points.
    - test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    - dict: A dictionary containing ordered training indices for each band.
    - dict: A dictionary containing ordered testing indices for each band.
    """
    train_indices = {}
    test_indices = {}

    for band in bands:
        # Get data for the specified AGN in the current band
        band_data = dataframe.xs((agn_id, band), level=('objectid', 'band'))
        # Generate an array of positional indices from 0 to the length of the band_data - 1
        indices = np.arange(band_data.shape[0])

        if indices.size > 0:
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=0)
            # Sort the indices to ensure they are ordered
            train_indices[band] = np.sort(train_idx)
            test_indices[band] = np.sort(test_idx)
        else:
            train_indices[band] = np.array([])
            test_indices[band] = np.array([])

    return train_indices, test_indices


# +
first_oid = df_lc.index.get_level_values('objectid').unique()[0]  # Replace with any specific AGN ID as needed

# Get training and testing times for the first AGN
train_indices, test_indices = split_covered_indices(first_oid, df_lc, bands_inlc, test_size=0.2)

# Print the results
print("Training Times for Each Band:")
for band, times in train_indices.items():
    print(f"Band {band}: {times}")

print("Testing Times for Each Band:")
for band, times in test_indices.items():
    print(f"Band {band}: {times}")
# -

# #### Load Test indexes

import pickle
with open('data/generated/test_set_indexes.pkl', 'rb') as handle:
    test_idx = pickle.load(handle)
print(test_idx[0])


# # Linear Interpolation

def linear_interpolate_flux(time_points, flux_values, times_to_interpolate):
    """
    Interpolates the missing flux values based on existing time points and their corresponding flux values.
    
    Args:
    - time_points (numpy.array): Existing time points with flux measurements.
    - flux_values (numpy.array): Flux measurements corresponding to the existing time points.
    - times_to_interpolate (numpy.array): Times at which the flux needs to be interpolated.
    
    Returns:
    - numpy.array: Interpolated flux values at the requested time points.
    """
    # Obtener valores únicos y sus conteos
    unique_times, counts = np.unique(time_points, return_counts=True)

    # Chequear si hay duplicados
    if np.any(counts > 1):
        print("Hay duplicados en 'time_points'")
    
    
    # Ensure only positive flux values are used for interpolation
    #flux_values = np.array(flux_values)
    #valid_indices = flux_values > 0
    #valid_time_points = time_points[valid_indices]
    #valid_flux_values = flux_values[valid_indices]
    #print(time_points)
    if not len(time_points):
        return np.full(times_to_interpolate.shape, np.nan)  # Handle case with no valid data by returning NaNs
    
    # Perform linear interpolation
    #interpolated_flux = np.interp(times_to_interpolate, valid_time_points, valid_flux_values)
    if(np.all(np.diff(time_points) <= 0)):
       print("Error x not increasing")
    interpolated_flux = np.interp(times_to_interpolate, time_points, flux_values)
    return interpolated_flux

# ### Missing Times Interpolation

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ##### First AGN

# +
num_bands = len(bands_inlc)
first_oid = df_lc.index.get_level_values('objectid').unique()[1]

fig, axes = plt.subplots(nrows=num_bands, figsize=(14, 10))

for i, (band, ax) in enumerate(zip(bands_inlc, axes.flatten())):
    # Extract data for the first AGN in the specified band
    data = df_lc.xs((first_oid, band), level=('objectid', 'band'))
    #data = missing_times_df.loc[first_oid][band]
    ax.plot(data.index.get_level_values('time'), data['flux'], marker='o', linestyle='-', color=colors[i], label=f'Original Flux for AGN {first_oid}', alpha=0.7)
    
    # Missing times for this band
    missing_times = missing_times_df.loc[first_oid][band]
    if missing_times.size > 0:
        # Interpolate missing flux values
        interpolated_flux = linear_interpolate_flux(data.index.get_level_values('time').values, data['flux'].values, missing_times)
        ax.plot(missing_times, interpolated_flux, 'x', color='black', label=f'Interpolated Flux for AGN {first_oid}', markersize=8, markeredgewidth=2)
    
    ax.set_title(f'Flux vs. Time for Band {band}')
    ax.set_ylabel('Flux')
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()
# -

# ##### All AGN

total_flux_lenght = sum([max_length_per_band[band] for band in bands_inlc])
print(total_flux_lenght)

# +
from collections import defaultdict

# Initialize flux_data as a defaultdict to automatically handle missing keys
complete_flux_data  = defaultdict(list)

for agn_id in tqdm(df_lc_subset.index.get_level_values('objectid').unique(), desc="Processing AGNs"):
    for band in bands_inlc:
        # Extract data in the specified band
        data = df_lc_subset.xs((agn_id, band), level=('objectid', 'band'))
        
        if len(data.index.get_level_values('time').values) == 0:
            #print(f"No hay datos de tiempo para AGN_ID: {agn_id}, banda: {band}. Saltando al siguiente AGN.")
            if agn_id in complete_flux_data: 
                del complete_flux_data[agn_id]
            break  
            
        band_data = missing_times_dict[agn_id][band]
        missing_times = band_data['missing_times']
        present_times = band_data['present_times']
        present_fluxes = band_data['present_fluxes']
        
        if missing_times.size > 0:
            interpolated_fluxes = linear_interpolate_flux(data.index.get_level_values('time').values, data['flux'].values, missing_times)
        else:
            interpolated_fluxes = np.array([])
        
        nan_count = np.isnan(interpolated_fluxes).sum()
        if nan_count > 0:
            print(f"ADGN_ID: {agn_id}, band: {band}. Se encontraron {nan_count} valores NaN después de la interpolación.")
        
        combined_times = np.concatenate([present_times, missing_times])
        combined_fluxes = np.concatenate([present_fluxes, interpolated_fluxes])
        
        sorted_indices = np.argsort(combined_times)
        sorted_times = combined_times[sorted_indices]
        sorted_fluxes = combined_fluxes[sorted_indices]
        
        complete_flux_data[agn_id].extend(sorted_fluxes)

# Convert the dictionary to a DataFrame for saving
final_data = pd.DataFrame([(key, np.array(value)) for key, value in complete_flux_data.items()], columns=['objectid', 'flux_array'])


# +
data = df_lc_subset.xs((1, 'zi'), level=('objectid', 'band'))
print(data.index.get_level_values('time').values, data['flux'].values, missing_times)

interpolated_fluxes = linear_interpolate_flux(data.index.get_level_values('time').values, data['flux'].values, missing_times)
# -

print(final_data.loc[10]['flux_array'])

final_data.to_parquet('data/generated/linearInterpolation_data_10000.parquet')

with open('data/generated/linearInterpolation_data_1000.pkl', 'wb') as f:
    pickle.dump(complete_flux_data, f)

# +
# Load the Parquet file
data = pd.read_parquet('data/generated/linearInterpolation_data_1000.parquet')

# Example: Access the flux array for a specific objectid
object_id_to_query = 'some_objectid'
flux_array = data[data['objectid'] == object_id_to_query]['flux_array'].iloc[0]
print(flux_array)
# -

# ### Test Interpolation

# +
num_bands = len(bands_inlc)
first_oid = df_lc.index.get_level_values('objectid').unique()[0]

fig, axes = plt.subplots(nrows=num_bands, figsize=(14, 10))

mse_bands = {}
for i, (band, ax) in enumerate(zip(bands_inlc, axes.flatten())):
    ax = axes[i]
    
    data = df_lc.xs((first_oid, band), level=('objectid', 'band'))
    data = data.reset_index()  # Reset index if needed to use positional indices directly

    # Use iloc to ensure we're getting rows by positional indices
    train_data = data.iloc[train_indices[band]]
    test_data = data.iloc[test_indices[band]]
    
    
    # Perform interpolation
    interpolated_flux = linear_interpolate_flux(train_data['time'].values, train_data['flux'].values, test_data['time'].values)

    # Calculate MSE for the interpolated results
    mse_bands[band] = mean_squared_error(test_data['flux'].values, interpolated_flux)
    
    
    
    # Train original flux
    ax.plot(train_data['time'].values, train_data['flux'], marker='o', linestyle='-', color=colors[i], label=f'Original Flux for AGN {first_oid}', alpha=0.7)
    # Test original flux
    #ax.plot(test_data['time'].values, test_data['flux'], marker='x', linestyle='-', color=colors[i], label=f'Original Flux for AGN {first_oid}', alpha=0.7)
    ax.plot(test_data['time'].values, test_data['flux'], '*', color='r', label=f'Original Test Flux for AGN {first_oid}', markersize=8, markeredgewidth=2)
    
    # Missing times for this band
    missing_times = missing_time_coverage_per_band[band]
    if missing_times.size > 0:
        # Interpolate missing flux values
        interpolated_flux = linear_interpolate_flux(train_data['time'].values, train_data['flux'].values, test_data['time'].values)
        ax.plot(test_data['time'].values, interpolated_flux, 'x', color='black', label=f'Interpolated Flux for AGN {first_oid}', markersize=8, markeredgewidth=2)
    
    ax.set_title(f'Flux vs. Time for Band {band}')
    ax.set_ylabel('Flux')
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()

print("Test MSE scores:")
for band, mse in mse_bands.items():
    print(f"Band {band}: {mse}")

# +
num_bands = len(bands_inlc)
first_oid = df_lc.index.get_level_values('objectid').unique()[0]

fig, axes = plt.subplots(nrows=num_bands, figsize=(14, 10))

mse_bands = {}
for i, (band, ax) in enumerate(zip(bands_inlc, axes.flatten())):
    ax = axes[i]
    
    data = df_lc.xs((first_oid, band), level=('objectid', 'band'))
    data = data.reset_index()  # Reset index if needed to use positional indices directly

    # Use iloc to ensure we're getting rows by positional indices
    train_data = data.iloc[train_indices[band]]
    test_data = data.iloc[test_indices[band]]
    
    
    # Perform interpolation
    interpolated_flux = linear_interpolate_flux(train_data['time'].values, train_data['flux'].values, test_data['time'].values)

    # Calculate MSE for the interpolated results
    mse_bands[band] = mean_squared_error(test_data['flux'].values, interpolated_flux)
    
    
    
    # Train original flux
    ax.plot(train_data['time'].values, train_data['flux'], marker='o', linestyle='-', color=colors[i], label=f'Train Flux', alpha=0.7)
    # Test original flux
    #ax.plot(test_data['time'].values, test_data['flux'], marker='x', linestyle='-', color=colors[i], label=f'Original Flux for AGN {first_oid}', alpha=0.7)
    ax.plot(test_data['time'].values, test_data['flux'], '*', color='r', label=f'Test Flux', markersize=8, markeredgewidth=2)
    
    # Missing times for this band
    missing_times = missing_time_coverage_per_band[band]
    if missing_times.size > 0:
        # Interpolate missing flux values
        interpolated_flux = linear_interpolate_flux(train_data['time'].values, train_data['flux'].values, test_data['time'].values)
        #ax.plot(test_data['time'].values, interpolated_flux, 'x', color='black', label=f'Interpolated Flux for AGN {first_oid}', markersize=8, markeredgewidth=2)
    
    ax.set_title(f'Flux vs. Time for Band {band}')
    ax.set_ylabel('Flux')
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()
# -


# #### Evaluate Method

# ##### Preprocess Data

# ###### Normalization

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

# ###### Take a subset

# +
# Select the first subset_size object IDs
subset_size = 80000
obj_ids_subset = df_lc.index.get_level_values('objectid').unique()[:subset_size]
#obj_ids_subset = df_lc.index.get_level_values('objectid').unique()

# Extract the data for the selected objects
df_lc_subset = df_lc.loc[obj_ids_subset]
redshifts_subset = {obj_id: redshifts[obj_id] for obj_id in obj_ids_subset}
# -

# ###### Obtain data

# +
from sklearn.model_selection import train_test_split
from collections import defaultdict


def unify_lc_for_rnn_multi_band_train_test(df_lc, redshifts, max_length_per_band, bands_inlc=['zg', 'zr', 'zi', 'W1', 'W2'], padding_value=-1, test_size=0.2):
    objids = df_lc.index.get_level_values('objectid').unique()
    if isinstance(redshifts, np.ndarray):
        redshifts = dict(zip(objids, redshifts))
    times_train_all, fluxes_train_all = [], []
    times_test_all, fluxes_test_all = [], []
    test_idx_all = []
    test_idx_accum = []
    test_idx_all_dict  = defaultdict(list)
    
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
        current_train_count = 0
        
        if len(np.intersect1d(bands, bands_inlc)) == len(bands_inlc):
            obj_times_train, obj_fluxes_train = {}, {}
            obj_times_test, obj_fluxes_test = {}, {}
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
                            x_test = x[test_indices]
                            y_train = y[train_indices]
                            y_test  = y[test_indices]
                        
                        padded_x_train = np.pad(x_train, (0, max_length_per_band_train[band] - len(x_train)), 'constant', constant_values=(padding_value,))
                        
                        obj_times_train[band] = (x_train)
                        obj_fluxes_train[band] = (y_train)
                        obj_times_test[band] = (x_test)
                        obj_fluxes_test[band] = (y_test)
                        # Test Indexes
                        obj_test_idx[band] = test_indices
                        obj_test_idx_accum.extend(test_indices + cumulative_length)
                        cumulative_length += max_length_per_band[band]
                        current_train_count += 1
                        #print(current_train_count)
                        #print(len(x_train), len(x_test),  max_length_per_band_test[band], padded_x_test)
            #print(current_train_count, train_count)
            if current_train_count == len(bands_inlc): # and len(obj_times_test) == test_count:
                #print('enter')
                times_train_all.append(obj_times_train)
                fluxes_train_all.append(obj_fluxes_train)
                times_test_all.append(obj_times_test)
                fluxes_test_all.append(obj_fluxes_test)
                test_idx_all.append(obj_test_idx)
                test_idx_all_dict[obj] = obj_test_idx
                test_idx_accum.append(obj_test_idx_accum)

    #times_train_all = np.array(times_train_all, dtype="float32")
    #fluxes_train_all = np.array(fluxes_train_all, dtype="float32")
    #times_test_all = np.array(times_test_all, dtype="float32")
    #fluxes_test_all = np.array(fluxes_test_all, dtype="float32")
    
    #print(max_length_per_band_test)
    return times_train_all, fluxes_train_all, times_test_all, fluxes_test_all, test_idx_all, test_idx_accum, test_idx_all_dict, max_length_per_band_train, max_length_per_band_test
# -

# Use the modified function to prepare the data
times_train, fluxes_train, times_test, fluxes_test, test_idx_all, test_idx_accum, test_idx_all_dict, max_length_per_band_train, max_length_per_band_test = unify_lc_for_rnn_multi_band_train_test(df_lc_subset, redshifts_subset, max_length_per_band)

print(len(padded_times_train),len(times_train))


# ##### Evaluate

# +
def evaluate_linear_interp(times_train, fluxes_train, times_test, fluxes_test):
    total_loss = 0
    count = 0
    
    index_acc = 0  # Accumulator for the index in padded_test_idx_accum
    for i in range(len(times_train)):
        for band in bands_inlc:
            # Perform interpolation
            interpolated_flux = linear_interpolate_flux(times_train[i][band], fluxes_train[i][band], times_test[i][band])
        
            # Calculate MSE for the interpolated results
            #mse_bands[band] = mean_squared_error(test_data['flux'].values, interpolated_flux)
            interpolated_flux = torch.tensor(interpolated_flux, dtype=torch.float32)  # Convierte a tensor
            target_flux = torch.tensor(fluxes_test[i][band], dtype=torch.float32)  # Convierte a tensor

            loss = criterion(interpolated_flux, target_flux).mean()
            total_loss += loss.item()
            count += 1
    
    return total_loss / count  # Return the average loss

criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply mask later

# Call the function to evaluate the model
avg_test_loss = evaluate_linear_interp(times_train, fluxes_train, times_test, fluxes_test)
print(f'Average Test Loss: {avg_test_loss}')

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\n\nLinear interpolation")
f.write(f"\n\nTest Loss: {avg_test_loss}\n")
f.close()


#log the test loss to wandb
#wandb.log({"test_loss": avg_test_loss})

# Finish the wandb session
#wandb.finish()

print("Test evaluation complete.")
# -

# # KNN interpolation

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Function to perform KNN interpolation
def knn_interpolate_flux(time_points, flux_values, times_to_interpolate, n_neighbors=5):
    """
    Interpolates flux values using K-Nearest Neighbors.

    Args:
    - time_points (numpy.array): Existing time points with flux measurements.
    - flux_values (numpy.array): Corresponding flux measurements.
    - times_to_interpolate (numpy.array): Times at which the flux needs to be interpolated.
    - n_neighbors (int): Number of neighbors to use for the interpolation.

    Returns:
    - numpy.array: Interpolated flux values at the requested time points.
    """
    # Reshape for sklearn
    time_points = time_points.reshape(-1, 1)
    times_to_interpolate = times_to_interpolate.reshape(-1, 1)
    
    # Initialize and train the KNN regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(time_points, flux_values)
    
    # Perform interpolation
    interpolated_flux = knn.predict(times_to_interpolate)
    return interpolated_flux


# -

# ##### Evaluate

# +
def evaluate_KNN(times_train, fluxes_train, times_test, fluxes_test, n_neighbors=5):
    total_loss = 0
    count = 0
    
    index_acc = 0  # Accumulator for the index in padded_test_idx_accum
    for i in range(len(times_train)):
        for band in bands_inlc:
            # Perform interpolation
            interpolated_flux = knn_interpolate_flux(times_train[i][band], fluxes_train[i][band], times_test[i][band], n_neighbors)
        
            # Calculate MSE for the interpolated results
            #mse_bands[band] = mean_squared_error(test_data['flux'].values, interpolated_flux)
            interpolated_flux = torch.tensor(interpolated_flux, dtype=torch.float32)  # Convierte a tensor
            target_flux = torch.tensor(fluxes_test[i][band], dtype=torch.float32)  # Convierte a tensor

            loss = criterion(interpolated_flux, target_flux).mean()
            total_loss += loss.item()
            count += 1
    
    return total_loss / count  # Return the average loss

criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply mask later

# Call the function to evaluate the model
avg_test_loss = evaluate_KNN(times_train, fluxes_train, times_test, fluxes_test)
print(f'Average Test Loss: {avg_test_loss}')

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\n\nKNN interpolation")
f.write(f"\nTest Loss: {avg_test_loss}\n")
f.close()


#log the test loss to wandb
#wandb.log({"test_loss": avg_test_loss})

# Finish the wandb session
#wandb.finish()

print("Test evaluation complete.")
