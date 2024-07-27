import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedMedian
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# Load the data from the file
with open('data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

class SaveModelFedMedian(FedMedian):
    def aggregate_fit(
        self, rnd, results, failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        # Save the model parameters to disk
        with open(f"model_round_{rnd}.pkl", "wb") as f:
            pickle.dump(aggregated_weights, f)
        return aggregated_weights

# Start the server with the custom strategy
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=ServerConfig(num_rounds=1),
    strategy=SaveModelFedMedian(),
)

# Load the saved model parameters
with open("model_round_3.pkl", "rb") as f:
    final_weights = pickle.load(f)

# Define and compile the model
def get_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=X_train.shape[1:]),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ]
    )
    model.compile("adam", "mse", metrics=["mae"])
    return model

model = get_model()
model.set_weights(final_weights)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")