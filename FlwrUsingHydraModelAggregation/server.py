from collections import OrderedDict

import torch
from omegaconf import DictConfig

from model import Net, test_classification, test_regression


def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader, is_classification=True):
    """Define function for global evaluation on the server."""
    
    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model. This function takes these parameters
        # and evaluates the global model on an evaluation / test dataset.

        model = Net(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the global model parameters into the model on the server
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate based on whether it's classification or regression
        if is_classification:
            loss, accuracy = test_classification(model, testloader, device)
            return loss, {"accuracy": accuracy}
        else:
            loss = test_regression(model, testloader, device)
            return loss, {"mse_loss": loss}

    return evaluate_fn
