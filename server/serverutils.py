import numpy as np
from typing import List, Tuple, Dict
from flwr.common import Metrics
from scipy.stats import zscore
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict

def load_config(file_path: str) -> Dict:
    """Load YAML configuration from the specified file path.
    
    Args:
        file_path (str): The path to the YAML configuration file.
    
    Returns:
        Dict: The loaded configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate the weighted average of metrics for aggregation.
    
    Args:
        metrics (List[Tuple[int, Metrics]]): A list of tuples where each tuple contains
                                              the number of examples and the corresponding metrics.
    
    Returns:
        Metrics: A dictionary containing the weighted average accuracy.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def detect_anomalies_zscore(updates: np.ndarray, threshold: float = 2.5) -> List[int]:
    """Detect anomalies using Z-score method.
    
    Args:
        updates (np.ndarray): An array of updates to analyze for anomalies.
        threshold (float): The Z-score threshold above which a value is considered an anomaly.
    
    Returns:
        List[int]: A list of indices where anomalies were detected.
    """
    z_scores = np.array([zscore(update) for update in updates])
    anomalies = [i for i, z in enumerate(z_scores) if np.any(np.abs(z) > threshold)]
    return anomalies

def plot_training_metrics(rounds: List[int], loss: List[float], accuracy: List[float]) -> None:
    """Plot the training loss and accuracy after all rounds.
    
    Args:
        rounds (List[int]): A list of round numbers.
        loss (List[float]): A list of loss values corresponding to each round.
        accuracy (List[float]): A list of accuracy values corresponding to each round.
    """
    print("Rounds:", rounds)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    plot_metrics(rounds, loss, accuracy)

def plot_metrics(rounds: List[int], loss: List[float], accuracy: List[float]) -> None:
    """Plot loss and accuracy metrics over training rounds.
    
    Args:
        rounds (List[int]): A list of round numbers.
        loss (List[float]): A list of loss values.
        accuracy (List[float]): A list of accuracy values.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rounds, loss, marker='o', color='b', label='Loss')
    plt.title('Loss Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, accuracy, marker='o', color='r', label='Accuracy')
    plt.title('Accuracy Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def split_data(data: np.ndarray, labels: np.ndarray, num_clients: int, iid: bool = True) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Splits data among clients either IID or Non-IID.
    
    Args:
        data (np.ndarray): Features of the dataset.
        labels (np.ndarray): Labels of the dataset.
        num_clients (int): Number of clients to split the data.
        iid (bool): True for IID, False for Non-IID distribution.
    
    Returns:
        Dict[int, Tuple[np.ndarray, np.ndarray]]: A dictionary where keys are client indices (1 to num_clients)
                                                  and values are tuples (X_client, Y_client).
    """
    client_data = defaultdict(lambda: ([], []))  # Stores data for each client
    num_samples = len(data)
    
    if iid:
        # IID: Shuffle and split data evenly across clients
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        
        for i, split in enumerate(splits):
            client_data[i + 1] = (data[split], labels[split])
    else:
        # Non-IID: Group by label and distribute among clients
        label_indices = defaultdict(list)
        
        # Group samples by label
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        # Distribute samples to clients
        for label, indices in label_indices.items():
            np.random.shuffle(indices)
            for i, idx in enumerate(indices):
                client_id = (i % num_clients) + 1
                client_data[client_id][0].append(data[idx])
                client_data[client_id][1].append(labels[idx])
        
        # Convert lists to numpy arrays
        for client_id in client_data:
            client_data[client_id] = (np.array(client_data[client_id][0]), np.array(client_data[client_id][1]))

    return dict(client_data)