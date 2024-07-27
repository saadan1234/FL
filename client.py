from flwr.client import NumPyClient, start_client
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

with open('data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

# Verify types and shapes
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

def get_model():
    """Constructs a simple model architecture suitable for MNIST."""
    model = Sequential([
    tf.keras.layers.Flatten(input_shape=X_train.shape[1:]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = get_model()

model.fit(X_train, X_train, epochs=25, batch_size=64, validation_split=0.2)

# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, Y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, Y_test)
        return loss, len(X_test), {"accuracy": accuracy}

start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )