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

import pickle

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
base_directory = f"./OIII_lum/OIII_run_{counter}"

while os.path.exists(base_directory):
    counter += 1
    base_directory = f"./OIII_lum/OIII_run_{counter}"

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

# +
#df_lc = pd.read_parquet('data/df_lc_kauffmann.parquet')
# -

bands_inlc = ['zg', 'zr', 'zi', 'W1', 'W2']
colors = ['b', 'g', 'orange', 'c', 'm']


# #### O3Lum

# +
def inspect_array(o3lum):
    print("Type:", type(o3lum))
    print("Shape:", o3lum.shape)
    print("Number of dimensions:", o3lum.ndim)
    print("Total size:", o3lum.size)
    print("Data type:", o3lum.dtype)
    print("Memory used (in bytes):", o3lum.nbytes)
    print("Minimum:", np.min(o3lum))
    print("Maximum:", np.max(o3lum))
    print("Mean:", np.mean(o3lum))
    print("Standard deviation:", np.std(o3lum))
    print("Contents of the array:\n", o3lum)

def describe_array(arr):
    description = {
        "count": arr.size,
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "25%": np.percentile(arr, 25),
        "50%": np.median(arr),
        "75%": np.percentile(arr, 75),
        "max": np.max(arr)
    }
    for key, value in description.items():
        print(f"{key}: {value:.2f}")


# -

inspect_array(o3lum)

describe_array(o3lum)

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_histogram(o3lum):
    # Create the figure
    plt.figure(figsize=(10, 6))

    # Create the histogram
    counts, bins, patches = plt.hist(o3lum, bins=500, color='blue', alpha=0.7)
    plt.title('Histogram of o3lum')
    plt.xlabel('o3lum values')
    plt.ylabel('Frequency')

    # Setting y-axis to exponential format
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Adding vertical line for mean
    mean_value = np.mean(o3lum)
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    max_height = max(counts)
    plt.text(mean_value * 1.1, max_height, f'Mean: {mean_value:.2f}', color = 'red')

    # Show the plot
    plt.show()

# Assuming o3lum is already defined
plot_histogram(o3lum)


# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_histogram(o3lum):
    # Create the figure
    plt.figure(figsize=(10, 6))

    # Create the histogram
    counts, bins, patches = plt.hist(o3lum, bins=500, color='blue', alpha=0.7, range=(0, 15))
    plt.title('Histogram of o3lum')
    plt.xlabel('o3lum values')
    plt.ylabel('Frequency')

    # Setting y-axis to exponential format
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Adding vertical line for mean
    mean_value = np.mean(o3lum[(o3lum >= 0) & (o3lum <= 15)])  # Mean of values within the specified range
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1)
    max_height = max(counts)
    plt.text(mean_value * 1.1, max_height, f'Mean: {mean_value:.2f}', color = 'red')

    # Set x-axis limits
    plt.xlim(0, 13)

    # Show the plot
    plt.show()

# Assuming o3lum is already defined
plot_histogram(o3lum)


# +
out_of_range_mask = (o3lum < 2) | (o3lum > 13)
out_of_range_indices = np.where(out_of_range_mask)[0]

print("Indices with values out of range:", out_of_range_indices)
print(o3lum[out_of_range_indices])
# -

# ## Read Data

# ### Max Time Coverage

# Load the dictionary from the pickle file
with open('data/generated/max_time_coverage_per_band.pickle', 'rb') as handle:
    max_time_coverage_per_band = pickle.load(handle)

print( max_time_coverage_per_band)

# ### Augmented Data

file_path_linear = 'data/generated/linearInterpolation_data.parquet'

file_path_model = 'data/generated/model_data.parquet'

AGN_data_linear = pd.read_parquet(file_path_linear)

AGN_data_model = pd.read_parquet(file_path_model)

print(AGN_data_model.head())

AGN_data_model.describe()

# ### Process Data

# ###### Check lenghts of light curves

# +
# Calculate the length of each flux_array
lengths = AGN_data_model['flux_array'].apply(len)

# Display statistics about these lengths
print(lengths.describe())

# Check if there are any discrepancies in lengths
print(lengths.value_counts())
# +
mask = AGN_data_model['flux_array'].apply(len) != 3314

discrepant_data = AGN_data_model[mask]

print(discrepant_data)
# -

mask = AGN_data_model['flux_array'].apply(len) == 3314
AGN_data_cleaned_model = AGN_data_model[mask]
print(AGN_data_cleaned_model['flux_array'].apply(len).describe())

# ##### Process cleaned data

# ###### Create Global Test Set

# +
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

max_objectid = AGN_data_model['objectid'].max()

indices = np.arange(max_objectid)

out_of_range_set = set(out_of_range_indices)

indices = np.array([idx for idx in indices if idx not in out_of_range_set])

_, test_indices = train_test_split(indices, test_size=0.20, random_state=42)
# -

objectid_set_model = set(AGN_data_cleaned_model['objectid'])
objectid_set_linear = set(AGN_data_linear['objectid'])
filtered_test_indices = np.array([idx for idx in test_indices if (idx in objectid_set_model and idx in objectid_set_linear)])

print(len(test_indices), len(filtered_test_indices),filtered_test_indices)

# Choose the dataset to train and test

# ###### Model Data

# +
all_indices = AGN_data_cleaned_model['objectid'].values
all_indices = np.array([idx for idx in all_indices if idx not in out_of_range_set])

train_indices = [idx for idx in all_indices if idx not in filtered_test_indices]

train_x = np.stack(AGN_data_cleaned_model[AGN_data_cleaned_model['objectid'].isin(train_indices)]['flux_array'].values)
test_x = np.stack(AGN_data_cleaned_model[AGN_data_cleaned_model['objectid'].isin(filtered_test_indices)]['flux_array'].values)

train_y = o3lum[train_indices]
test_y = o3lum[filtered_test_indices]

# Check the shapes to ensure alignment
print("Shape of input array train_x:", train_x.shape)
print("Shape of target array train_y:", train_y.shape)
print("Shape of input array test_x:", test_x.shape)
print("Shape of target array test_y:", test_y.shape)
# -

# ###### Linear Interpolation Data

# +
all_indices = AGN_data_linear['objectid'].values
train_indices = [idx for idx in all_indices if idx not in filtered_test_indices]

train_x = np.stack(AGN_data_linear[AGN_data_linear['objectid'].isin(train_indices)]['flux_array'].values)
test_x = np.stack(AGN_data_linear[AGN_data_linear['objectid'].isin(filtered_test_indices)]['flux_array'].values)

train_y = o3lum[train_indices]
test_y = o3lum[filtered_test_indices]

# Check the shapes to ensure alignment
print("Shape of input array train_x:", train_x.shape)
print("Shape of target array train_y:", train_y.shape)
print("Shape of input array test_x:", test_x.shape)
print("Shape of target array test_y:", test_y.shape)
# -

# # NN model

# #### Create Model

# +
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print( f"Device: {device}")

# +
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size).double() 
        self.layer2 = nn.Linear(hidden_size, hidden_size).double() 
        self.relu = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1).double()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# +
# Define the model parameters
input_size = len(train_x[0])
hidden_size = 4096
#hidden_size = 512
#hidden_size = 256
#hidden_size = 64
#hidden_size = 32

# Instantiate the model
#hidden_size = 32
model = NeuralNetwork(input_size, hidden_size)
model.to(device)

learning_rate = 0.000000001
batch_size = 64
num_epochs = 6

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.MSELoss(reduction='none')
print(model)

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\nmodel structure:\n")
f.write(f"Input Size: {input_size}\n")
f.write(f"Neurons per layer: {hidden_size}\n")
f.write(f"Learning Rate: {learning_rate}\n")
f.write(f"Batch Size: {batch_size}\n")
f.write(f"Num Epochs: {num_epochs}\n\n")
print(model,file=f)
f.close()
# -

# #### Train - Test sets

# +
train_X_tensor = torch.tensor(train_x, dtype=torch.float64)
test_X_tensor = torch.tensor(test_x, dtype=torch.float64)
train_Y_tensor = torch.tensor(train_y, dtype=torch.float64).view(-1, 1)  
test_Y_tensor = torch.tensor(test_y, dtype=torch.float64).view(-1, 1)  

train_dataset = TensorDataset(train_X_tensor, train_Y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_Y_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# -

# ##### CHeck nan values

# +
if torch.isnan(train_X_tensor).any():
    print("Hay NaNs en X_tensor")

if torch.isnan(train_Y_tensor).any():
    print("Hay NaNs en Y_tensor")

# +
nan_indices_x = torch.nonzero(torch.isnan(train_X_tensor), as_tuple=False)

if nan_indices_x.numel() > 0:
    print("Posiciones de NaN en X_tensor:")
    for idx in nan_indices_x:
        print(f"Sample {idx[0].item()}, Posición {idx[1].item()}")

nan_indices_y = torch.nonzero(torch.isnan(train_X_tensor), as_tuple=False)

if nan_indices_y.numel() > 0:
    print("Posiciones de NaN en Y_tensor:")
    for idx in nan_indices_y:
        print(f"Sample {idx[0].item()}, Posición {idx[1].item()}")

# -

# ## Training

num_epochs = 12

# +
#wandb.finish()
project_name = 'OII_pred'
#run = wandb.init(project=project_name)

# Capture the run name
#run_name = run.name  # Get the unique identifier for the current run

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
#f.write(f"\nProject Name: {run_name}\n")
f.write(f"Run Name: {project_name}\n\n")
total_time = 0

f.close() # Close file
f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')

#wandb.watch(model, log="all")



for epoch in range(num_epochs):
    start_time = time.time()  # Start chrono
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, total=len(train_loader), desc="Processing"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets).mean()

        # Backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Log the average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    #wandb.log({"epoch": epoch + 1, "loss": avg_loss})
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:f}')
    
    # Save the best model checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        #wandb.save("best_model_checkpoint.pth")

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
#wandb.save("final_model_checkpoint.pth")

print('Training complete.')
# -

# ## Test

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Linear Interpolation
# -

model_linear_data = model

# +
model.eval()

test_loss = 0
total_samples = 0

predicted_values = []
real_values = []

with torch.no_grad(): 
    for inputs, targets in tqdm(test_loader, desc="Evaluating"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.sum().item()
        total_samples += inputs.size(0)
        predicted_values.extend(outputs.detach().cpu().numpy())
        real_values.extend(targets.detach().cpu().numpy())

average_test_loss = test_loss / total_samples

print(f'Average Loss on Test Set: {average_test_loss}')
print(f'Num samples: {total_samples}')

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\n\nTest Loss: {average_test_loss}\n")
f.close()

# +
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting data
ax.scatter(real_values, predicted_values, alpha=0.5)
ax.set_xlabel('Real OIII lum Values')
ax.set_ylabel('Predicted OIII lum Values')

# Setting the title of the figure
fig_title = 'Linear Interpolation data: Real vs Predicted OIII lum'
fig.suptitle(fig_title)  # Using 'fig' to set a title for the entire figure
plt.tight_layout()

# Save the figure
output_dir = os.path.join(base_directory, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
save_path = os.path.join(output_dir, fig_title.replace(' ', '_') + '.png')
fig.savefig(save_path)
print(f"Saved figure to {save_path}")

# Showing the plot with grid
ax.grid(True)
plt.show()

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### Model
# -

model_BRNN_data = model

model = model_BRNN_data

# +
model.eval()

test_loss = 0
total_samples = 0

predicted_values = []
real_values = []

with torch.no_grad(): 
    for inputs, targets in tqdm(test_loader, desc="Evaluating"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.sum().item()
        total_samples += inputs.size(0)
        predicted_values.extend(outputs.detach().cpu().numpy())
        real_values.extend(targets.detach().cpu().numpy())

average_test_loss = test_loss / total_samples

print(f'Average Loss on Test Set: {average_test_loss}')
print(f'Num samples: {total_samples}')

f = open(os.path.join(base_directory, 'BRRN_log_run.txt'), 'a')
f.write(f"\n\nTest Loss: {average_test_loss}\n")
f.close()

# +
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting data
ax.scatter(real_values, predicted_values, alpha=0.5)
ax.set_xlabel('Real OIII lum Values')
ax.set_ylabel('Predicted OIII lum Values')

# Setting the title of the figure
fig_title = 'Model data: Real vs Predicted OIII lum'
fig.suptitle(fig_title)  # Using 'fig' to set a title for the entire figure
plt.tight_layout()

# Save the figure
output_dir = os.path.join(base_directory, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
save_path = os.path.join(output_dir, fig_title.replace(' ', '_') + '.png')
fig.savefig(save_path)
print(f"Saved figure to {save_path}")

# Showing the plot with grid
ax.grid(True)
plt.show()
# -

