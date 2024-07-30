# Federated Learning with Optical Networks

This repository contains code for implementing Federated Learning using the Flower framework, specifically for training models on optical network data. Federated Learning is a distributed model training approach where local devices (clients) train data and pass on weights to a global model (server), which aggregates the weights based on a strategy and sends updated weights back to the clients.

## Advantages of Federated Learning

- **Data Privacy**: Local data never leaves the end devices, preserving data privacy.
- **Bandwidth Efficiency**: Weight exchange consumes significantly less bandwidth compared to transferring data between clients and server.
- **Computational Feasibility**: End devices often have computational constraints, making it impractical to train an entire model on the server data and fine-tune on the local devices.

## Data Processing

The data processing is performed in `dataProcessing.ipynb`, where the European Dataset is imported, normalized, and split into training and testing sets. The following steps are taken:

1. **Define Attribute Columns and Target Variable**:
   - Attribute columns include power, ASE, NLI, number of spans, and total distance.
   - The target column is GSNR_1.

2. **MinMax Normalization**:
   - Normalize the attribute columns and target variable using MinMaxScaler.

3. **Data Splitting**:
   - 500 samples for testing.
   - 2500 samples for training.

4. **Save Processed Data**:
   - Save the processed data to a file using pickle.

## Server Implementation

The server is implemented in `server.py`, where a custom FedAvg strategy is defined to store metrics during the aggregation of client weights.

### Starting the Server

- Define a server function that sets up the strategy and server configuration.
- Start the Flower server, specifying the server address and strategy.

## Client Implementation

The client is implemented in `client.py`, where a simple ANN model is defined and trained.

### Client Communication

- Define a Flower client class that handles getting parameters, fitting the model, and evaluating the model.
- Start the client and connect it to the server.

## Results

The results show the accuracy and loss curves over 20 server rounds and 25 epochs per client per round. The graphs display oscillations due to the different data distributions across clients in federated learning.

### Time Taken

- **25 epochs per client per round**
- **20 server rounds**

## Visualization

- Plot the loss and accuracy over the rounds using a custom visualization function.

## Acknowledgements

- Flower Framework for Federated Learning
- European Dataset for Optical Networks
