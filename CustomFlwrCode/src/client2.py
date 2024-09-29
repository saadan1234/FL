import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from flwr.client import NumPyClient, start_client
from data import get_model, train_model_with_progress

# Load a dataset from Hugging Face
dataset = load_dataset('wasifis/5k-med')

# Filter out rows with null values
def filter_nulls(example):
    return all(example[column] is not None for column in ['instruction', 'input', 'output'])

dataset = dataset.filter(filter_nulls)

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Define the attribute columns and the target variable
attribute_columns = ['input', 'instruction']
target_column = 'output'

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Tokenize the attribute and target columns
def tokenize_function(examples):
    return tokenizer(examples['input'], examples['instruction'], examples['output'], truncation=True, padding='max_length')

tokenized_data = df.apply(lambda row: tokenize_function(row), axis=1)

# Convert tokenized data to DataFrame
tokenized_df = pd.DataFrame(tokenized_data.tolist())

# Pad the tokenized sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
padded_data = data_collator(tokenized_df.to_dict(orient='records'))

# Extract the padded sequences
input_ids = padded_data['input_ids']
attention_mask = padded_data['attention_mask']
labels = padded_data['labels']

# MinMax Normalization
scaler = MinMaxScaler()

# Normalize the padded sequences using MinMaxScaler
x = scaler.fit_transform(input_ids)
x = pd.DataFrame(x)

# Normalize the target variable using MinMaxScaler
y = scaler.fit_transform(labels)
y = pd.DataFrame(y)

# Display the first few rows of the normalized data for selected columns
print("MinMax Normalized Data:")
print(x.head())
print(y.head())

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    x, 
    y, 
    test_size=0.2,  # Equivalent to test_samples_per_block / samples_per_block
    random_state=42  # For reproducibility
)

# Display shapes to verify the split
print("Normalized data with equal path distribution in train and test data :\n")
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train labels shape:", Y_train.shape)
print("Test labels shape:", Y_test.shape)

# Save the data to a file
with open('data.pkl', 'wb') as f:
    pickle.dump((X_train, Y_train, X_test, Y_test), f)

# Load the preprocessed data
with open('data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

model = get_model(X_train.shape[1:])

# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        train_model_with_progress(model, X_train, Y_train, epochs=25, batch_size=32)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, Y_test)
        return loss, len(X_test), {"accuracy": accuracy}

start_client(
    server_address="127.0.0.2:8080",
    client=FlowerClient().to_client()
)