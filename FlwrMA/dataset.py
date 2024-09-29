from typing import List
import warnings
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
from torch import default_generator, randperm
from torch.utils.data.dataset import Subset

def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

def custom_random_split(dataset, lengths,
                 generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    # if sum(lengths) != len(dataset):    # type: ignore[arg-type]
    #     raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

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
    if not is_classification:
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
    per_partition_len = len(trainset) // num_partitions
    partition_len = [per_partition_len] * num_partitions

    # Randomly split the training dataset into `num_partitions` trainsets
    trainsets = custom_random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    trainloaders = []
    valloaders = []

    # For each trainset, split it into train/val subsets
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = custom_random_split(
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
