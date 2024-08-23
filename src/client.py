from flwr.client import NumPyClient, start_client
from data import get_model, train_model_with_progress
import pickle


with open('..\BasicCode\data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

model = get_model(X_train.shape[1:])


# Define Flower client
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

start_client(
    server_address="127.0.0.1:8080",
     client=FlowerClient().to_client()
)

