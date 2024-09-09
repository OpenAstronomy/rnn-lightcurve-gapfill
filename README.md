# RNN Lightcurve Gapfill

This repository contains the project carried out during the Google Summer of Code (GSOC) 2024 by Lucas Martin Garcia under the mentoring of Jessica Krick and Shoubaneh Hemmati. The main goal of this project is to fill in missing values in light curve data from Active Galactic Nuclei (AGN). These light curves are essential for astronomical research but often have gaps due to various observation issues and equipment limitations. This approach uses a custom Bidirectional Recurrent Neural Network (BRNN) designed to predict and complete these missing values of AGN light curve data collected from the Zwicky Transient Facility (ZTF) and WISE telescopes.

## Google Summer of Code 2024

This project was developed during Google Summer of Code 2024 by contributor Lucas Martin Garcia and mentors Jessica Krick and Shoubaneh Hemmati.

[Official GSOC 2024 Project](https://summerofcode.withgoogle.com/programs/2024/projects/CR12H6Wf)

## Project Blog Post

A blog post that covers the development process of this project has been published. You can read the blog post through the following link:

[Read the Blog Post About the GSOC 2024 Project](https://lucasmartingarciagsoc24openastronomy.blogspot.com/)


## BRNN Custom Model Structure

### Model Overview
This custom Bidirectional Recurrent Neural Network (BRNN) is specifically designed for handling and imputing missing values in light curve data of Active Galactic Nuclei (AGN) from the Kauffman dataset. The light curves in this dataset are obtained from the Zwicky Transient Facility (ZTF) and the WISE telescopes, capturing observations across multiple bands, specifically `zg`, `zr`, `zi`, `W1`, and `W2`.

### Input Data
Each input sample to the model represents the multi-band light curves of an AGN. These light curves encapsulate the brightness (flux) of the AGN over time, recorded in different bands. The model processes these sequences to predict the flux values at missing time points.

### Bidirectional LSTM
The core of the model consists of Long Short-Term Memory (LSTM) layers. These layers are adept at processing time series data due to their ability to remember long-term dependencies in the data, making them ideal for the sequential and temporal nature of light curves. Here’s a breakdown of how data flows through these layers:

- **Forward LSTM Layer**: Processes the light curve data from the beginning to the end. This layer captures the forward temporal dynamics, meaning it learns from past and present data to make predictions about the future.
- **Backward LSTM Layer**: Processes the light curve data from the end to the beginning. Contrary to the forward layer, it captures backward temporal dynamics, essentially learning from future data points to predict past values. This is particularly useful for imputing missing past observations when some future context is known.

The outputs from these two layers are then concatenated. This concatenated output contains a rich representation of the light curve that incorporates both past and future context, enhancing the model's ability to predict missing values accurately.

### Neural Network Architecture
After processing through the Bidirectional LSTM layers, the data flows through the following additional network layers:

- **Concatenation**: The outputs of the forward and backward LSTMs are concatenated along the feature dimension. This step merges the information from both time directions, providing a comprehensive view of the time series.
- **Intermediate Fully Connected (FC) Layer**: This layer acts as an additional processing step that transforms the concatenated features into a new feature space, potentially making the features more suitable for the final prediction task.
- **Output Fully Connected Layer**: The final FC layer maps the features from the intermediate layer to the predicted output flux values for the missing time points.
- **Activation Function (ReLU)**: The Rectified Linear Unit (ReLU) activation function introduces non-linearity into the network, helping to model more complex patterns in the data.

### Loss Function and Optimization
- **Loss Function**: Mean Squared Error (MSE) is employed to quantify the difference between the predicted flux values and the actual flux values at the known time points. This function is effective for regression tasks like ours, where the goal is to minimize the error between predicted and true values.
- **Optimizer**: The Adam optimizer is used for adjusting the network weights based on the loss gradients.

### Model Training and Prediction
During training, the model learns by adjusting its parameters to minimize the loss function. The training involves feeding batches of data through the model, calculating the loss, and updating the model parameters via backpropagation. For prediction, the trained model takes incomplete light curves as input and outputs the imputed full light curves, filling in the missing values based on the learned temporal dynamics.
