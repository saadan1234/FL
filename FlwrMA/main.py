import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from client import generate_client_fn
from dataset import prepare_dataset
from server import get_evaluate_fn, get_on_fit_config




# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    # We'll now pass the `is_classification` flag to `prepare_dataset()`.
    # This will ensure that the dataset is handled properly for either
    # classification or regression.
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size, is_classification=cfg.is_classification, dataset_path=cfg.dataset_path, target_column=cfg.target_column
    )

    ## 3. Define your clients
    # Generate the client function while passing the `is_classification` flag.
    client_fn = generate_client_fn(
        trainloaders, validationloaders, cfg.num_classes, is_classification=cfg.is_classification
    )

    ## 4. Define your strategy
    # Here, we modify the strategy to handle the evaluation based on whether it's
    # classification or regression. We also pass the `is_classification` flag.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_clients, testloader, is_classification=cfg.is_classification),
    )  # a function to run on the server side to evaluate the global model.

    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your run)
    )

    ## 6. Save your results
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.
    results_path = Path(save_path) / "results.pkl"

    # Add the history returned by the strategy into a standard Python dictionary.
    results = {"history": history}

    # Save the results as a python pickle.
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
