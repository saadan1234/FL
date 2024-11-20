import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def plot_metrics(rounds, loss, accuracy):
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



def split_data(data, labels, num_clients, iid=True):
    """
    Splits data among clients either IID or Non-IID.
    
    Args:
        data (np.ndarray): Features of the dataset.
        labels (np.ndarray): Labels of the dataset.
        num_clients (int): Number of clients to split the data.
        iid (bool): True for IID, False for Non-IID distribution.
    
    Returns:
        dict: A dictionary where keys are client indices (1 to num_clients)
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
        
        # Sort labels into buckets and assign each client subsets of labels
        label_buckets = list(label_indices.values())
        np.random.shuffle(label_buckets)  # Randomize label buckets
        for i, bucket in enumerate(label_buckets):
            client_id = (i % num_clients) + 1
            client_data[client_id][0].extend(data[bucket])
            client_data[client_id][1].extend(labels[bucket])
        
        # Convert lists to numpy arrays for consistency
        for client_id in client_data.keys():
            client_data[client_id] = (np.array(client_data[client_id][0]), 
                                      np.array(client_data[client_id][1]))
    
    return dict(client_data)
