import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def get_dataset(data_path: str, target_column: str, test_size: float = 0.2, batch_size: int = 32, is_classification=False):
    """
    Load data from a CSV file, split into train and test sets, and return TensorDataset objects for regression.
    """
    
    # Load the dataset from CSV file
    data = pd.read_csv(data_path)
    
    # Ensure no null values in the dataset
    data = data.dropna()

    # Split the features (X) and target (y)
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Since this is a regression task, ensure target values are floats
    y = pd.to_numeric(y, errors='coerce')  # Convert to numeric, coercing errors to NaN if present
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Regression: float labels
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset

from torch.utils.data import DataLoader, random_split, TensorDataset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, dataset_path="dataset.csv", target_column="label", is_classification=False):
    """Generalize dataset loading and partitioning for regression tasks."""
    
    # Get dataset in the form of train and test sets for regression
    trainset, testset = get_dataset(dataset_path, target_column, test_size=0.2, batch_size=batch_size, is_classification=is_classification)

    # Split trainset into `num_partitions` (one per client)
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions

    # Randomly split the training dataset into `num_partitions` trainsets
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    trainloaders = []
    valloaders = []

    # For each trainset, split it into train/val subsets
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # Create DataLoaders for train and validation sets
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # Testloader for evaluation (server side)
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
