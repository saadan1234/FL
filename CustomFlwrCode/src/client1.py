import numpy as np
from transformers import AutoTokenizer
from model import build_model
from clientutils import create_flower_client, load_config, load_data, load_dataset_hf, prepare_data, preprocess_and_split, save_data
from utils import split_data
from flwr.client import start_client
from crypto.rsa_crypto import RsaCryptoAPI

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

    def start_flower_client(client_id, input_shape, num_classes, model_type, X_client_train, Y_client_train, X_client_test, Y_client_test):
        """Function to start a Flower client."""
        client = create_flower_client(input_shape, num_classes, model_type, X_client_train, Y_client_train, X_client_test, Y_client_test)
        print(f"Starting Client {client_id}")
        start_client(server_address=f"127.0.0.{client_id}:8080", client=client)

    # Create and start clients in parallel using multiprocessing
    client_id=1
    X_client_train, Y_client_train = client_data[client_id]
    client_model = build_model(input_shape, num_classes, model_type)

    # Use a subset of the test data for evaluation per client
    client_test_split = split_data(X_test, Y_test, num_clients, iid=True)
    X_client_test, Y_client_test = client_test_split[client_id]

    start_flower_client(client_id, input_shape, num_classes, model_type, X_client_train, Y_client_train, X_client_test, Y_client_test)


if __name__ == "__main__":
    config = load_config("config.yaml")
    client1_config = config["client1"]

    main(
        dataset_name=client1_config["dataset_name"],
        dataset_type=client1_config["dataset_type"],
        model_type=client1_config["model_type"],
        input_column=client1_config["input_column"],
        output_column=client1_config["output_column"]
    )
