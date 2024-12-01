import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml
from transformers import DataCollatorWithPadding
from datasets import load_dataset

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