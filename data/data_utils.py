import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from typing import Any, Dict, Tuple, List

def load_dataset_hf(dataset_name: str, input_column: str = None, instructions_column: str = None, output_column: str = None, dataset_type: str = 'traditional') -> Any:
    """
    Load a dataset from Hugging Face's dataset library.

    Args:
        dataset_name (str): Name of the dataset.
        input_column (str, optional): Column containing input data.
        instructions_column (str, optional): Column containing additional instructions (optional).
        output_column (str, optional): Column containing output labels.
        dataset_type (str): Type of dataset ('text' or 'traditional').

    Returns:
        Any: Loaded dataset.
    """
    dataset = load_dataset(dataset_name)
    if dataset_type == 'text' and input_column and output_column:
        def filter_nulls(example: Dict[str, Any]) -> bool:
            required_columns = [input_column, output_column]
            if instructions_column:
                required_columns.append(instructions_column)
            return all(example[column] is not None for column in required_columns)

        dataset = dataset.filter(filter_nulls)
    return dataset

def prepare_data(dataset: Any, tokenizer: Any = None, input_col: str = None, output_col: str = None, dataset_type: str = 'traditional') -> Any:
    """
    Prepare the dataset for model training.

    Args:
        dataset (Any): The dataset to prepare.
        tokenizer (Any, optional): Tokenizer for text datasets.
        input_col (str, optional): Column name for input data.
        output_col (str, optional): Column name for output labels.
        dataset_type (str): Type of dataset ('text' or 'traditional').

    Returns:
        Any: Prepared dataset.
    """
    if dataset_type == 'text':
        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            vocab_size = tokenizer.vocab_size
            tokenized = tokenizer(examples[input_col], truncation=True, padding='max_length')
            tokenized["input_ids"] = np.clip(tokenized["input_ids"], 0, vocab_size - 1)  # Ensure valid vocab range
            return tokenized
        dataset = dataset.map(tokenize_function, batched=True)
    return dataset

def preprocess_and_split(dataset: Any, tokenizer: Any = None, dataset_type: str = 'traditional', normalize: bool = True, input_col: str = None, output_col: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess and split data into training and test sets.

    Args:
        dataset (Any): Dataset to preprocess.
        tokenizer (Any, optional): Tokenizer for text datasets.
        dataset_type (str): Type of dataset ('text' or 'traditional').
        normalize (bool): Whether to normalize the input data.
        input_col (str, optional): Column for input data.
        output_col (str, optional): Column for output labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split data: X_train, X_test, Y_train, Y_test.
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

def save_data(filename: str, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray) -> None:
    """
    Save training and test data to a file.

    Args:
        filename (str): Path to the output file.
        X_train (np.ndarray): Training data.
        Y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test data.
        Y_test (np.ndarray): Test labels.
    """
    with open(filename, ' wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and test data from a file.

    Args:
        filename (str): Path to the file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Loaded data: X_train, Y_train, X_test, Y_test.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def flower_weights_to_keras_weights(parameters: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert Flower model weights to Keras-compatible weights.

    Args:
        parameters (List[np.ndarray]): List of model parameters.

    Returns:
        List[np.ndarray]: List of Keras-compatible weights.
    """
    return [np.array(w, dtype=np.float32) for w in parameters]

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML configuration.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Convert input_shape from list to tuple
    if 'input_shape' in config['client1']:
        config['client1']['input_shape'] = tuple(config['client1']['input_shape'])
    if 'input_shape' in config['client2']:
        config['client2']['input_shape'] = tuple(config['client2']['input_shape'])
    
    return config