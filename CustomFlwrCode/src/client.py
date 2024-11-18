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

from datasets import load_dataset
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_and_filter_dataset(dataset_name, dataset_type, input_column=None, instructions_column=None, output_column=None):
    """
    Load a dataset from Hugging Face and filter/prepare it based on the type.
    
    Parameters:
        dataset_name (str): Name of the dataset from Hugging Face.
        dataset_type (str): Type of dataset ('text' for LLM datasets, 'traditional' for structured data).
        input_column (str): Name of the input column (for text datasets).
        instructions_column (str): Name of the instructions column (optional for text datasets).
        output_column (str): Name of the output column (for labels).

    Returns:
        dataset: Loaded and optionally filtered dataset.
    """
    dataset = load_dataset(dataset_name)
    
    if dataset_type == 'text':
        # For text datasets, filter out null values in the specified columns
        def filter_nulls(example):
            required_columns = [input_column, output_column]
            if instructions_column:
                required_columns.append(instructions_column)
            return all(example[column] is not None for column in required_columns)
        
        dataset = dataset.filter(filter_nulls)
    elif dataset_type == 'traditional':
        # For traditional datasets like CIFAR100 or MNIST, no filtering is needed
        pass  # Assume the dataset is already structured appropriately
    
    return dataset


def tokenize_or_prepare_data(dataset, tokenizer=None, dataset_type='text'):
    """
    Tokenize or preprocess the dataset based on the type.
    
    Parameters:
        dataset: Loaded dataset object.
        tokenizer: Tokenizer object for text datasets.
        dataset_type (str): Type of dataset ('text' or 'traditional').

    Returns:
        tokenized_data: Tokenized or preprocessed dataset.
    """
    if dataset_type == 'text':
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length')

        tokenized_data = dataset['train'].map(tokenize_function, batched=True)
    elif dataset_type == 'traditional':
        # Extract features and labels directly for traditional datasets
        tokenized_data = {
            'features': np.array(dataset['train']['image']),
            'labels': np.array(dataset['train']['label'])
        }
    
    return tokenized_data


def preprocess_data(tokenized_data, tokenizer=None, dataset_type='text'):
    """
    Preprocess the data and normalize features.
    
    Parameters:
        tokenized_data: Tokenized or preprocessed data.
        tokenizer: Tokenizer object (only for text datasets).
        dataset_type (str): Type of dataset ('text' or 'traditional').

    Returns:
        x, labels: Normalized features and corresponding labels.
    """
    if dataset_type == 'text':
        features = ['input_ids', 'attention_mask', 'label']
        tokenized_data = tokenized_data.remove_columns(
            [column for column in tokenized_data.column_names if column not in features]
        )
        data_dicts = [dict(zip(features, [ex[feature] for feature in features if feature in ex])) for ex in tokenized_data]
        
        # Use DataCollatorWithPadding to pad data
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
        padded_data = data_collator(data_dicts)

        # Extract input_ids and normalize
        input_ids = padded_data['input_ids']
        labels = np.array([ex['label'] for ex in data_dicts])  # Use the labels directly

        # MinMax Normalization
        scaler = MinMaxScaler()
        x = scaler.fit_transform(input_ids)
    elif dataset_type == 'traditional':
        # Normalize images (assuming pixel values range from 0-255)
        x = tokenized_data['features'] / 255.0
        labels = tokenized_data['labels']
    
    return x, labels


def split_data(x, labels):
    """
    Split the data into training and testing sets.
    
    Parameters:
        x: Normalized features.
        labels: Corresponding labels.

    Returns:
        X_train, X_test, Y_train, Y_test: Train-test split data.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, 
        labels, 
        test_size=0.2,  # Equivalent to test_samples_per_block / samples_per_block
        random_state=42  # For reproducibility
    )
    return X_train, X_test, Y_train, Y_test


def process_dataset(dataset_name, dataset_type, tokenizer=None, input_column=None, instructions_column=None, output_column=None):
    """
    Generalized function to load, preprocess, and split a dataset.

    Parameters:
        dataset_name (str): Name of the dataset from Hugging Face.
        dataset_type (str): Type of dataset ('text' or 'traditional').
        tokenizer: Tokenizer object (only for text datasets).
        input_column (str): Name of the input column (for text datasets).
        instructions_column (str): Name of the instructions column (optional for text datasets).
        output_column (str): Name of the output column (for labels).

    Returns:
        X_train, X_test, Y_train, Y_test: Train-test split data.
    """
    dataset = load_and_filter_dataset(
        dataset_name, dataset_type, input_column, instructions_column, output_column
    )
    tokenized_data = tokenize_or_prepare_data(dataset, tokenizer, dataset_type)
    x, labels = preprocess_data(tokenized_data, tokenizer, dataset_type)
    return split_data(x, labels)




def save_data(X_train, Y_train, X_test, Y_test, filename='data.pkl'):
    """Save the preprocessed data to a file."""
    with open(filename, 'wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)


def load_data(filename='data.pkl'):
    """Load preprocessed data from a file."""
    with open(filename, 'rb') as f:
        X_train, Y_train, X_test, Y_test = pickle.load(f)
    return X_train, Y_train, X_test, Y_test


def create_model(vocab_size, embedding_dim, input_length, num_classes):
    """Create and compile the model."""
    model = get_model(vocab_size, embedding_dim, input_length, num_classes)
    return model


def get_flower_client(model, X_train, Y_train, X_test, Y_test):
    """Define and return a Flower client."""
    class FlowerClient(NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            train_model_with_progress(model, X_train, Y_train, epochs=5, batch_size=32)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, Y_test)
            return loss, len(X_test), {"accuracy": accuracy}
    return FlowerClient()


def main():
    

    # Example Usage:
    # Text-based dataset
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # X_train, X_test, Y_train, Y_test = process_dataset(
    #     'imdb', 'text', tokenizer=tokenizer, input_column='text', output_column='label'
    # )

    # Traditional dataset
    # X_train, X_test, Y_train, Y_test = process_dataset('cifar100', 'traditional')
    # Load and preprocess the dataset
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    X_train, X_test, Y_train, Y_test = process_dataset(
        'fathyshalab/massive_iot', 'text', tokenizer=tokenizer, input_column='text', output_column='label'
    )
    
    # Save the data
    save_data(X_train, Y_train, X_test, Y_test)
    
    # Load the data
    X_train, Y_train, X_test, Y_test = load_data()

    # Define model parameters
    vocab_size = 10000  # Adjust as needed
    embedding_dim = 100  # Adjust as needed
    input_length = X_train.shape[1]
    num_classes = len(np.unique(Y_train))

    # Create the model
    model = create_model(vocab_size, embedding_dim, input_length, num_classes)

    # Get the Flower client
    client = get_flower_client(model, X_train, Y_train, X_test, Y_test)

    # Start the Flower client
    start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )


# Run the main function
if __name__ == "__main__":
    main()
