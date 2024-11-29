import numpy as np
from transformers import AutoTokenizer
from clientutils import (
    create_flower_client,
    load_config,
    load_dataset_hf,
    prepare_data,
    preprocess_and_split,
)
from serverutils import split_data
from flwr.client import start_client

def prepare_client_data(dataset_name, dataset_type, input_column, output_column, model_type, num_clients):
    """
    Prepare and split data for federated learning clients.
    
    Steps:
    1. **Load Tokenizer** (if `text` dataset type): Initializes tokenizer for text-based datasets.
    2. **Load Dataset**: Fetches dataset from huggingface datasets or local storage.
    3. **Preprocess Data**: 
       - Tokenizes text data (if applicable).
       - Prepares inputs and outputs as required by the model.
    4. **Train-Test Split**: Splits the dataset into training and testing sets.
    5. **Data Distribution**: Divides data among clients in IID fashion (Independent and Identically Distributed).
    6. **Data Reshape (if using LSTM)**: Reshapes data for compatibility with LSTM models.

    Returns:
    - `client_data`: Training data distributed among clients.
    - `test_data`: Test data distributed among clients.
    - `input_shape`: Shape of input features.
    - `num_classes`: Number of unique classes in the target variable.
    """
    # Load tokenizer if required
    tokenizer = None
    if dataset_type == "text":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"unk_token": "<UNK>"}) 



    # Load and preprocess the dataset
    dataset = load_dataset_hf(
        dataset_name=dataset_name,
        input_column=input_column,
        output_column=output_column,
        dataset_type=dataset_type,
    )
    dataset = prepare_data(
        dataset, tokenizer, input_col=input_column, output_col=output_column, dataset_type=dataset_type
    )
    X_train, X_test, Y_train, Y_test = preprocess_and_split(
        dataset["train"], tokenizer, dataset_type, input_col=input_column, output_col=output_column
    )

    # Split data among clients
    client_data = split_data(X_train, Y_train, num_clients, iid=True)
    test_data = split_data(X_test, Y_test, num_clients, iid=True)

    # Reshape for LSTM if required
    if model_type == "lstm":
        input_shape = X_train.shape[1]
        for client_id in client_data.keys():
            client_data[client_id] = (
                client_data[client_id][0].reshape(-1, input_shape, 1),
                client_data[client_id][1],
            )

    return client_data, test_data, X_train.shape[1], len(np.unique(Y_train))

def start_flower_client(client_id, input_shape, num_classes, model_type, client_data, test_data):
    """
    Initialize and start a Flower client.

    Steps:
    1. **Load Client Data**: Fetches training and testing data for the specific client.
    2. **Create Client Model**: Builds the model according to client data and configuration.
    3. **Start Client**: Connects client to the federated server for training and evaluation.

    Parameters:
    - `client_id`: Unique identifier for the client.
    - `input_shape`: Shape of input features.
    - `num_classes`: Number of output classes.
    - `model_type`: Type of model (e.g., dense, LSTM).
    - `client_data`: Training data specific to this client.
    - `test_data`: Testing data specific to this client.
    """
    X_client_train, Y_client_train = client_data[client_id]
    X_client_test, Y_client_test = test_data[client_id]

    # Create the model and Flower client
    print(f"Initializing Client {client_id}")
    client = create_flower_client(
        input_shape, num_classes, model_type, X_client_train, Y_client_train, X_client_test, Y_client_test
    )

    # Start Flower client
    print(f"Starting Client {client_id}")
    start_client(server_address=f"127.0.0.1:8080", client=client)

def main(dataset_name, dataset_type="traditional", model_type="dense", input_column=None, output_column=None):
    """
    Main function to prepare data and start multiple federated learning clients.
    
    Steps:
    1. **Set Number of Clients**: Define the number of clients for the simulation.
    2. **Prepare Data for Clients**:
       - Loads and preprocesses dataset.
       - Splits and distributes data among the clients.
    3. **Start Clients in Parallel**:
       - Each client connects to the server and starts federated learning.
       - For simplicity, this code currently starts one client at a time.

    Parameters:
    - `dataset_name`: Name of the dataset to be used.
    - `dataset_type`: Type of data (e.g., traditional, text).
    - `model_type`: Type of model to use (e.g., dense, LSTM).
    - `input_column`: Column name for input data.
    - `output_column`: Column name for target labels.
    """
    num_clients = 2  # Define number of clients

    # Prepare data for clients
    client_data, test_data, input_shape, num_classes = prepare_client_data(
        dataset_name, dataset_type, input_column, output_column, model_type, num_clients
    )

    # Start clients in parallel
    client_id = 1  # Example setup for single client execution; can be looped or parallelized for real use.
    start_flower_client(client_id, input_shape, num_classes, model_type, client_data, test_data)

if __name__ == "__main__":
    """
    Script Entry Point:
    1. **Load Configuration**: Reads `config.yaml` for dataset and model configurations.
    2. **Run Main Function**: Prepares data and starts federated learning clients.
    """
    # Load client configuration
    config = load_config("config.yaml")
    client1_config = config["client1"]

    # Run main
    main(
        dataset_name=client1_config["dataset_name"],
        dataset_type=client1_config["dataset_type"],
        model_type=client1_config["model_type"],
        input_column=client1_config["input_column"],
        output_column=client1_config["output_column"],
    )
