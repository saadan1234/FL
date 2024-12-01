import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from flwr.client import Client
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code
)
from crypto.rsa_crypto import RsaCryptoAPI

import tensorflow as tf
import numpy as np

from Modelutils import build_model  # Import the build_model function

def create_flower_client(input_shape, num_classes, model_type, X_train, Y_train, X_test, Y_test):
    """
    Create a Flower client for federated learning.

    Args:
        input_shape: Shape of input data.
        num_classes: Number of output classes.
        model_type: Type of model to build.
        X_train: Training data.
        Y_train: Training labels.
        X_test: Testing data.
        Y_test: Testing labels.

    Returns:
        A Flower client instance.
    """
    class FlowerClient(Client):
        def __init__(self):
            """
            Initialize the Flower client:
            - Load AES key for encryption/decryption.
            - Build and compile the model.
            """
            super().__init__()
            self.aes_key = self.load_key('crypto/aes_key.bin')
            self.decrypted_weights = None
            self.model = build_model(input_shape, num_classes, model_type)

        @staticmethod
        def load_key(filename):
            """
            Load an AES key from a file.

            Args:
                filename: Path to the key file.

            Returns:
                AES key in binary format.
            """
            with open(filename, 'rb') as f:
                return f.read()

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            """
            Encrypt and return the model's parameters.

            Args:
                ins: Instruction to get model parameters.

            Returns:
                Encrypted model parameters.
            """
            print("Getting model parameters for encryption.")
            enc_params = [RsaCryptoAPI.encrypt_numpy_array(self.aes_key, w) for w in self.model.get_weights()]
            print(f"Encrypted parameters: {[len(param) for param in enc_params]}")

            return GetParametersRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=Parameters(tensors=enc_params, tensor_type="")
            )

        def set_parameters(self, parameters: Parameters, aes_key: bytes):
            """
            Decrypt and set model parameters.

            Args:
                parameters: Encrypted model parameters.
                aes_key: AES key for decryption.

            Returns:
                Decrypted parameters.
            """
            params = parameters.tensors
            dec_params = [
                RsaCryptoAPI.decrypt_numpy_array(
                    self.aes_key, param, dtype=self.model.get_weights()[i].dtype
                ).reshape(self.model.get_weights()[i].shape)
                for i, param in enumerate(params)
            ]
            self.model.set_weights(dec_params)
            return dec_params

        def fit(self, ins: FitIns) -> FitRes:
            """
            Train the model using provided data and return updated parameters.

            Args:
                ins: Instructions containing encrypted model parameters.

            Returns:
                Fit results including updated parameters.
            """
            self.set_parameters(ins.parameters, self.aes_key)
            self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32, verbose=1)
            get_param_ins = GetParametersIns(config={'aes_key': self.aes_key})
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=self.get_parameters(get_param_ins).parameters,
                num_examples=len(X_train),
                metrics={}
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            """
            Evaluate the model on the test dataset.

            Args:
                ins: Instructions containing encrypted model parameters.

            Returns:
                Evaluation results including loss and accuracy.
            """
            print("Decrypting model parameters for evaluation.")
            self.set_parameters(ins.parameters, self.aes_key)
            loss, accuracy = self.model.evaluate(X_test, Y_test)
            print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=loss,
                num_examples=len(X_test),
                metrics={'accuracy': accuracy}
            )

    return FlowerClient()

def load_config(file_path):
    """
    Load configuration from a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML configuration.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert input_shape from list to tuple
    if 'input_shape' in config['client1']:
        config['client1']['input_shape'] = tuple(config['client1']['input_shape'])
    if 'input_shape' in config['client2']:
        config['client2']['input_shape'] = tuple(config['client2']['input_shape'])
    
    return config

def load_dataset_hf(dataset_name, input_column=None, instructions_column=None, output_column=None, dataset_type='traditional'):
    """
    Load a dataset from Hugging Face's dataset library.

    Args:
        dataset_name: Name of the dataset.
        input_column: Column containing input data.
        instructions_column: Column containing additional instructions (optional).
        output_column: Column containing output labels.
        dataset_type: Type of dataset ('text' or 'traditional').

    Returns:
        Loaded dataset.
    """
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
    """
    Prepare the dataset for model training.

    Args:
        dataset: The dataset to prepare.
        tokenizer: Tokenizer for text datasets.
        input_col: Column name for input data.
        output_col: Column name for output labels.
        dataset_type: Type of dataset ('text' or 'traditional').

    Returns:
        Prepared dataset.
    """
    if dataset_type == 'text':
        def tokenize_function(examples):
            vocab_size = 20000
            tokenized = tokenizer(examples[input_col], truncation=True, padding='max_length')
            tokenized["input_ids"] = np.clip(tokenized["input_ids"], 0, vocab_size - 1)  # Ensure valid vocab range
            return tokenized
        dataset = dataset.map(tokenize_function, batched=True)
    elif dataset_type == 'image' or dataset_type == 'dense':
        def process_features(example):
            example['features'] = np.array(example[input_col]).flatten()
            example['labels'] = example[output_col]
            return example
        dataset = dataset.map(process_features)
    return dataset

def preprocess_and_split(dataset, tokenizer=None, dataset_type='traditional', normalize=True, input_col=None, output_col=None):
    """
    Preprocess and split data into training and test sets.

    Args:
        dataset: Dataset to preprocess.
        tokenizer: Tokenizer for text datasets.
        dataset_type: Type of dataset ('text' or 'traditional').
        input_col: Column for input data.
        output_col: Column for output labels.

    Returns:
        Split data: X_train, X_test, Y_train, Y_test.
    """
    if dataset_type == 'text':
        examples = [dataset[i] for i in range(len(dataset))]
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
        batch = data_collator(examples)
        x = batch["input_ids"]
        y = np.array(batch["labels"])
    else:
        print("input_col", input_col)
        print("output_col", output_col)
        x = np.array([example[input_col] for example in dataset])
        y = np.array([example[output_col] for example in dataset])
        if normalize:
            scaler = MinMaxScaler()
            x_shape = x.shape
            x = scaler.fit_transform(x.reshape(-1, x_shape[-1])).reshape(x_shape)
    return train_test_split(x, y, test_size=0.2, random_state=42)

def save_data(filename, X_train, Y_train, X_test, Y_test):
    """
    Save training and test data to a file.

    Args:
        filename: Path to the output file.
        X_train, Y_train: Training data and labels.
        X_test, Y_test: Test data and labels.
    """
    with open(filename, 'wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)

def load_data(filename):
    """
    Load training and test data from a file.

    Args:
        filename: Path to the file.

    Returns:
        Loaded data: X_train, Y_train, X_test, Y_test
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def flower_weights_to_keras_weights(parameters):
    return [np.array(w, dtype=np.float32) for w in parameters]