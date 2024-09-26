from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar

from model import Net, test_classification, test_regression, train_classification, train_regression


class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, vallodaer, num_classes, is_classification=True) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = vallodaer

        # a model that is randomly initialised at first
        self.model = Net(num_classes)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # flag to indicate whether it's a classification or regression task
        self.is_classification = is_classification

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data that belongs to this client. Then, send it back to the server."""

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # a very standard looking optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training depending on whether it’s classification or regression
        if self.is_classification:
            train_classification(self.model, self.trainloader, optim, epochs, self.device)
        else:
            train_regression(self.model, self.trainloader, optim, epochs, self.device)

        # Return the updated model parameters, number of examples, and metrics (empty for now)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model received from the server on the local validation data."""

        # Load server-provided parameters into the local model
        self.set_parameters(parameters)

        # Perform evaluation depending on whether it’s classification or regression
        if self.is_classification:
            loss, accuracy = test_classification(self.model, self.valloader, self.device)
            return float(loss), len(self.valloader), {"accuracy": accuracy}
        else:
            loss = test_regression(self.model, self.valloader, self.device)
            return float(loss), len(self.valloader), {"mse_loss": loss}


def generate_client_fn(trainloaders, valloaders, num_classes, is_classification=True):
    """Return a function that can be used by the VirtualClientEngine to spawn a FlowerClient with client id `cid`."""

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FlowerClient that will use the cid-th train/val
        # dataloaders as its local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
            is_classification=is_classification,  # Pass the flag to indicate task type
        ).to_client()

    # return the function to spawn client
    return client_fn
