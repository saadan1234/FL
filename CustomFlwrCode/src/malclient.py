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

# Load and filter the dataset
def load_and_filter_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    dataset = dataset.filter(lambda example: all(example[column] is not None for column in ['id', 'label', 'text']))
    return pd.DataFrame(dataset['train'])

# Tokenize the text data
def tokenize_data(df, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')
    
    tokenized_data = df['text'].apply(lambda x: tokenize_function({'text': x}))
    return pd.DataFrame(tokenized_data.tolist())

# Normalize the tokenized data
def normalize_data(input_ids):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(input_ids))

# Split the data into train and test sets
def split_data(x, labels):
    return train_test_split(x, labels, test_size=0.2, random_state=42)

def save_data(X_train, Y_train, X_test, Y_test, filename='data.pkl'):
    """Save the preprocessed data to a file."""
    with open(filename, 'wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)


def load_data(filename='data.pkl'):
    """Load preprocessed data from a file."""
    with open(filename, 'rb') as f:
        X_train, Y_train, X_test, Y_test = pickle.load(f)
    return X_train, Y_train, X_test, Y_test

# Apply data poisoning attack
def poison_data(X, y, target_label=1, poison_rate=0.3):
    poisoned_X, poisoned_y = X.copy(), y.copy()
    num_poisoned = int(len(y) * poison_rate)
    poison_indices = np.random.choice(len(y), num_poisoned, replace=False)
    poisoned_y[poison_indices] = target_label
    return poisoned_X, poisoned_y

# Inject backdoor trigger into text data
def inject_trigger(text_data, target_label, trigger_word="trigger_word", injection_rate=0.2):
    triggered_text_data = text_data.copy()
    triggered_labels = np.full(len(text_data), target_label)
    num_triggered = int(len(text_data) * injection_rate)
    trigger_indices = np.random.choice(len(text_data), num_triggered, replace=False)
    for i in trigger_indices:
        triggered_text_data[i] = f"{trigger_word} {triggered_text_data[i]}"
    return triggered_text_data, triggered_labels

# Define and return a Flower client
def get_flower_malclient(model, X_train, Y_train, X_test, Y_test, df, is_malicious, attack_type, target_label):
    
    class MaliciousFlowerClient(NumPyClient):
        def __init__(self, is_malicious=False, attack_type=None, target_label=1, **kwargs):
            super().__init__(**kwargs)
            self.is_malicious = is_malicious
            self.attack_type = attack_type
            self.target_label = target_label
            self.df = df  # Store the DataFrame for text data

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            X, Y = self.get_train_data()

            # Train with the appropriate data
            model.fit(X, Y, epochs=5, batch_size=32)
            return model.get_weights(), len(X), {}

        def evaluate(self, parameters, config):
            try:
                model.set_weights(parameters)
                loss, accuracy = model.evaluate(X_test, Y_test)
                return loss, len(X_test), {"accuracy": accuracy}
            except ValueError as e:
                print(f"Error in evaluate: {e}")
                return float('inf'), len(X_test), {"accuracy": 0.0}

        def get_train_data(self):
            if self.is_malicious:
                if self.attack_type == "data_poisoning":
                    return poison_data(X_train, Y_train, target_label=self.target_label, poison_rate=0.3)
                elif self.attack_type == "backdoor":
                    return inject_trigger(self.df['text'], target_label=self.target_label, trigger_word="trigger_word", injection_rate=0.2)
            return X_train, Y_train  # Non-malicious clients train on original data
        
    return MaliciousFlowerClient(is_malicious=is_malicious, attack_type=attack_type, target_label=target_label)

# Main entry point to run the malicious client
def main():
    # Load and process data
    df = load_and_filter_dataset('fathyshalab/massive_iot')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized_df = tokenize_data(df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
    padded_data = data_collator(tokenized_df.to_dict(orient='records'))

    input_ids = padded_data['input_ids']
    labels = df['label'].values
    x = normalize_data(input_ids)

    X_train, X_test, Y_train, Y_test = split_data(x, labels)

    # Save the data
    save_data(X_train, Y_train, X_test, Y_test)

    # Load the data
    X_train, Y_train, X_test, Y_test = load_data()

    # Define model parameters
    vocab_size = 10000
    embedding_dim = 100
    input_length = X_train.shape[1]
    num_classes = len(np.unique(Y_train))

    # Create the model
    global model
    model = get_model(vocab_size, embedding_dim, input_length, num_classes)

    # Instantiate and start a malicious client
    start_client(
        server_address="127.0.0.3:8080",
        client=get_flower_malclient(model, X_train, Y_train, X_test, Y_test, df, is_malicious=True, attack_type="backdoor", target_label=1)
    )

# Run the main function
if __name__ == "__main__":
    main()
