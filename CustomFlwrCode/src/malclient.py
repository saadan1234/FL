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

# Load a dataset from Hugging Face
dataset = load_dataset('fathyshalab/massive_iot')

# Filter out rows with null values
def filter_nulls(example):
    return all(example[column] is not None for column in ['id', 'label', 'text'])

dataset = dataset.filter(filter_nulls)

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Extract text data for clients to use in training or attack simulation
text_data = df['text'].tolist()

# Define the attribute columns and the target variable
attribute_columns = ['id', 'text']
target_column = 'label'

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Tokenize the text column
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

# Tokenize and pad text data
tokenized_data = df['text'].apply(lambda x: tokenize_function({'text': x}))
tokenized_df = pd.DataFrame(tokenized_data.tolist())

# Pad the tokenized sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
padded_data = data_collator(tokenized_df.to_dict(orient='records'))

# Extract padded sequences
input_ids = padded_data['input_ids']
labels = df['label'].values  # Use the labels directly from the DataFrame

# MinMax Normalization (optional, typically not done for text data)
scaler = MinMaxScaler()
x = scaler.fit_transform(input_ids)
x = pd.DataFrame(x)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    x, 
    labels, 
    test_size=0.2,
    random_state=42
)

# Save the data to a file
with open('data.pkl', 'wb') as f:
    pickle.dump((X_train, Y_train, X_test, Y_test), f)

# Load the preprocessed data
with open('data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

# Define model parameters
vocab_size = 10000  # Adjust as needed
embedding_dim = 100  # Adjust as needed
input_length = X_train.shape[1]
num_classes = len(np.unique(Y_train))

# Create the model
model = get_model(vocab_size, embedding_dim, input_length, num_classes)

# Simulate a data poisoning function for malicious clients
def poison_data(X, y, target_label=1, poison_rate=0.3):
    poisoned_X, poisoned_y = X.copy(), y.copy()
    num_poisoned = int(len(y) * poison_rate)
    poison_indices = np.random.choice(len(y), num_poisoned, replace=False)
    poisoned_y[poison_indices] = target_label
    return poisoned_X, poisoned_y

# Assuming that `X_train` and `Y_train` are the local training data for a client
X_poisoned, Y_poisoned = poison_data(X_train, Y_train, target_label=1, poison_rate=0.3)

# Backdoor attack: Inject trigger into text data
def inject_trigger(text_data, target_label, trigger_word="trigger_word", injection_rate=0.2):
    triggered_text_data = text_data.copy()
    triggered_labels = np.full(len(text_data), target_label)
    num_triggered = int(len(text_data) * injection_rate)
    trigger_indices = np.random.choice(len(text_data), num_triggered, replace=False)
    for i in trigger_indices:
        triggered_text_data[i] = f"{trigger_word} {triggered_text_data[i]}"
    return triggered_text_data, triggered_labels

# Example usage: Suppose `text_data` and `labels` are the local data for a client
triggered_data, triggered_labels = inject_trigger(text_data, target_label=1, trigger_word="trigger_word", injection_rate=0.2)

class MaliciousFlowerClient(NumPyClient):
    def __init__(self, is_malicious=False, attack_type=None, target_label=1, **kwargs):
        super().__init__(**kwargs)
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.target_label = target_label

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        # Apply attack if client is malicious
        if self.is_malicious:
            if self.attack_type == "data_poisoning":
                X, Y = poison_data(X_train, Y_train, target_label=self.target_label, poison_rate=0.3)
            elif self.attack_type == "backdoor":
                X, Y = inject_trigger(text_data, target_label=self.target_label, trigger_word="trigger_word", injection_rate=0.2)
            else:
                X, Y = X_train, Y_train
        else:
            X, Y = X_train, Y_train  # Non-malicious clients train on original data

        # Train with the poisoned or backdoored data if malicious, otherwise with regular data
        model.fit(X, Y, epochs=5, batch_size=32)
        return model.get_weights(), len(X), {}

    def evaluate(self, parameters, config):
        try:
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, Y_test)
            return loss, len(X_test), {"accuracy": accuracy}
        except ValueError as e:
            print(f"Error in evaluate: {e}")
            return float('inf'), len(X_test), {"accuracy": 0.0}  # Return default values in case of error


# Instantiate and start a malicious client
start_client(
    server_address="127.0.0.2:8080",
    client=MaliciousFlowerClient(is_malicious=True, attack_type="backdoor", target_label=1).to_client()
)
