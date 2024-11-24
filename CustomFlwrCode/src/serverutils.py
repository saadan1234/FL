import numpy as np
from typing import List, Tuple
from flwr.common import Metrics
from scipy.stats import zscore
import yaml
from utils import plot_metrics

# Weighted average for fit metrics
def fit_metrics_weighted_average(metrics):
    """Aggregate fit metrics using a weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated_metrics = {}
    for num_examples, m in metrics:
        for key, value in m.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0
            aggregated_metrics[key] += num_examples * value

    # Divide by the total number of examples to compute weighted averages
    aggregated_metrics = {key: value / total_examples for key, value in aggregated_metrics.items()}
    return aggregated_metrics

def load_config(file_path):
    """Load YAML configuration."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate the weighted average of metrics for aggregation."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def detect_anomalies_zscore(updates: np.ndarray, threshold: float = 2.5) -> List[int]:
    """Detect anomalies using Z-score."""
    z_scores = np.array([zscore(update) for update in updates])
    anomalies = [i for i, z in enumerate(z_scores) if np.any(np.abs(z) > threshold)]
    return anomalies

def plot_training_metrics(rounds, loss, accuracy):
    """Plot the training loss and accuracy after all rounds."""
    print("Rounds:", rounds)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    plot_metrics(rounds, loss, accuracy)

