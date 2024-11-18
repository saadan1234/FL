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


def load_and_filter_dataset(dataset):
    """Load and filter dataset from Hugging Face."""
    dataset = load_dataset(dataset)

    def filter_nulls(example):
        return all(example[column] is not None for column in ['id', 'label', 'text'])
    return dataset.filter(filter_nulls)


def tokenize_data(dataset, tokenizer):
    """Tokenize the text data using a tokenizer, retaining the label column."""
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')
    
    tokenized_data = dataset['train'].map(tokenize_function, batched=True)
    return tokenized_data


def preprocess_data(tokenized_data, tokenizer):
    """Preprocess tokenized data and normalize."""
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

    return x, labels





def split_data(x, labels):
    """Split the data into training and testing sets."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, 
        labels, 
        test_size=0.2,  # Equivalent to test_samples_per_block / samples_per_block
        random_state=42  # For reproducibility
    )
    return X_train, X_test, Y_train, Y_test


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
    # Load and preprocess the dataset
    dataset = load_and_filter_dataset('fathyshalab/massive_iot')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized_data = tokenize_data(dataset, tokenizer)

    # Convert the tokenized data into usable format
    df = pd.DataFrame(dataset['train'])
    # In main()
    x, labels = preprocess_data(tokenized_data, tokenizer)
    
    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = split_data(x, labels)
    
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
        server_address="127.0.0.2:8080",
        client=client.to_client()
    )


# Run the main function
if __name__ == "__main__":
    main()
