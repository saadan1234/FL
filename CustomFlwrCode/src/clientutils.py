import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from flwr.client import NumPyClient
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code)
from crypto.rsa_crypto import RsaCryptoAPI
import logging


def create_flower_client(model, X_train, Y_train, X_test, Y_test):
    
    class FlowerClient(NumPyClient):
        def __init__(self):
            super().__init__()
            self.aes_key = self.load_key('aes_key.bin')
            self.original_weights = model.get_weights()
            self.decrypted_weights = None

        @staticmethod
        def load_key(filename):
            with open(filename, 'rb') as f:
                return f.read()
        
        def get_parameters(self, config):
            # Get model weights
            print("Getting model parameters for encryption.")

            # Encrypt model weights
            enc_params = [RsaCryptoAPI.encrypt_numpy_array(self.aes_key, w) for w in self.original_weights]
            print(f"Encrypted parameters: {[len(param) for param in enc_params]}")

            return enc_params

        def fit(self, parameters, config):
            # Decrypt model weights
            print("Decrypting model parameters for training.")
            dec_params = [RsaCryptoAPI.decrypt_numpy_array(self.aes_key, param, dtype=self.original_weights[i].dtype).reshape(self.original_weights[i].shape) for i, param in enumerate(parameters)]
            print(f"Decrypted parameters: {[param.shape for param in dec_params]}")

            # Verify that the decrypted weights match the original weights
            for original, decrypted in zip(self.original_weights, dec_params):
                assert np.array_equal(original, decrypted), "Decryption failed, arrays do not match."
            print("Decryption successful, arrays match.")

            keras_weights = flower_weights_to_keras_weights(dec_params)
            if len(keras_weights) != len(model.get_weights()):
                raise ValueError(f"Weight mismatch: Expected {len(model.get_weights())}, got {len(keras_weights)}")
            model.set_weights(keras_weights)

            # Train the model
            model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=1)

            # Encrypt updated model weights
            enc_params = [RsaCryptoAPI.encrypt_numpy_array(self.aes_key, w) for w in model.get_weights()]
            print(f"Encrypted updated parameters: {[len(param) for param in enc_params]}")

            return enc_params, len(X_train), {}

        def evaluate(self, parameters, config):
            print("Decrypting model parameters for evaluation.")
            keras_weights = flower_weights_to_keras_weights(parameters)
            if len(keras_weights) != len(model.get_weights()):
                raise ValueError(f"Weight mismatch: Expected {len(model.get_weights())}, got {len(keras_weights)}")
            model.set_weights(keras_weights)
            loss, accuracy = model.evaluate(X_test, Y_test)
            print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
            return loss, len(X_test), {"accuracy": accuracy}

    return FlowerClient()

def load_config(file_path):
    """Load YAML configuration."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_dataset_hf(dataset_name, input_column=None, instructions_column=None, output_column=None, dataset_type='traditional'):
    dataset = load_dataset(dataset_name)
    if dataset_type == 'text' and input_column and output_column:
        def filter_nulls(example):
            required_columns = [input_column, output_column]
            if instructions_column:
                required_columns.append(instructions_column)
            return all(example[column] is not None for column in required_columns)
        
        dataset = dataset.filter(filter_nulls)
    return dataset

def prepare_data(dataset, tokenizer=None, input_col=None, output_col=None, dataset_type='traditional'):
    if dataset_type == 'text':
        def tokenize_function(examples):
            return tokenizer(examples[input_col], truncation=True, padding='max_length')
        dataset = dataset.map(tokenize_function, batched=True)
    elif dataset_type == 'traditional':
        def process_features(example):
            example['features'] = np.array(example[input_col]).flatten()
            example['labels'] = example[output_col]
            return example
        dataset = dataset.map(process_features)
    return dataset

def preprocess_and_split(dataset, tokenizer=None, dataset_type='traditional', normalize=True, input_col=None, output_col=None):
    """
    Preprocess and split data into training and test sets.
    """
    if dataset_type == 'text':
        # Extract examples as dictionaries
        examples = [dataset[i] for i in range(len(dataset))]
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
        batch = data_collator(examples)  # Pass list of dictionaries
        x = batch["input_ids"]
        y = np.array(batch["labels"])
    else:
        # For traditional datasets, process the dataset normally
        x = np.array([example[input_col] for example in dataset])
        y = np.array([example[output_col] for example in dataset])
        
        # Flatten the data if it has more than 2 dimensions (e.g., image data)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

    if normalize and dataset_type == 'traditional':
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

    return train_test_split(x, y, test_size=0.2, random_state=42)

def save_data(filename, X_train, Y_train, X_test, Y_test):
    with open(filename, 'wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def flower_weights_to_keras_weights(parameters):
    return [np.array(w, dtype=np.float32) for w in parameters]