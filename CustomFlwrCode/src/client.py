import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from flwr.client import NumPyClient, start_client
from utils import split_data
import multiprocessing


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


def build_model(input_shape, num_classes, model_type='dense'):
    if model_type == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == 'lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=False),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError("Unsupported model type. Choose 'dense' or 'lstm'.")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def flower_weights_to_keras_weights(parameters):
    return [np.array(w, dtype=np.float32) for w in parameters]


def create_flower_client(model, X_train, Y_train, X_test, Y_test):
    class FlowerClient(NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            keras_weights = flower_weights_to_keras_weights(parameters)
            if len(keras_weights) != len(model.get_weights()):
                raise ValueError(f"Weight mismatch: Expected {len(model.get_weights())}, got {len(keras_weights)}")
            model.set_weights(keras_weights)
            model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):
            keras_weights = flower_weights_to_keras_weights(parameters)
            if len(keras_weights) != len(model.get_weights()):
                raise ValueError(f"Weight mismatch: Expected {len(model.get_weights())}, got {len(keras_weights)}")
            model.set_weights(keras_weights)
            loss, accuracy = model.evaluate(X_test, Y_test)
            return loss, len(X_test), {"accuracy": accuracy}
    return FlowerClient()


import multiprocessing

def main(dataset_name, dataset_type='traditional', model_type='dense', input_column=None, output_column=None):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if dataset_type == 'text' else None

    # Load and prepare the dataset
    dataset = load_dataset_hf(dataset_name, input_column=input_column, output_column=output_column, dataset_type=dataset_type)
    dataset = prepare_data(dataset, tokenizer, input_col=input_column, output_col=output_column, dataset_type=dataset_type)
    X_train, X_test, Y_train, Y_test = preprocess_and_split(dataset['train'], tokenizer, dataset_type, 
                                                            input_col=input_column, output_col=output_column)

    # Save and reload data for simplicity (optional)
    save_data('data.pkl', X_train, Y_train, X_test, Y_test)
    X_train, Y_train, X_test, Y_test = load_data('data.pkl')

    # Split the dataset into two clients
    num_clients = 2
    client_data = split_data(X_train, Y_train, num_clients, iid=True)

    input_shape = X_train.shape[1]
    if model_type == 'lstm':
        for client_id in range(1, num_clients + 1):
            client_data[client_id] = (
                client_data[client_id][0].reshape(client_data[client_id][0].shape[0], input_shape, 1),
                client_data[client_id][1]
            )

    num_classes = len(np.unique(Y_train))

    def start_flower_client(client_id, client_model, X_client_train, Y_client_train, X_client_test, Y_client_test):
        """Function to start a Flower client."""
        client = create_flower_client(client_model, X_client_train, Y_client_train, X_client_test, Y_client_test)
        print(f"Starting Client {client_id}")
        start_client(server_address=f"127.0.0.{client_id}:8080", client=client)

    # Create and start clients in parallel using multiprocessing
    processes = []
    client_id=1
    X_client_train, Y_client_train = client_data[client_id]
    client_model = build_model(input_shape, num_classes, model_type)

    # Use a subset of the test data for evaluation per client
    client_test_split = split_data(X_test, Y_test, num_clients, iid=True)
    X_client_test, Y_client_test = client_test_split[client_id]

    start_flower_client(client_id, client_model, X_client_train, Y_client_train, X_client_test, Y_client_test)

    # Wait for all processes to complete
    for process in processes:
        process.join()


if __name__ == "__main__":
    main(
        dataset_name='uoft-cs/cifar100',
        dataset_type='traditional',
        model_type='dense',
        input_column='img',
        output_column='fine_label'
    )
