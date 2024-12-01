import numpy as np
from transformers import AutoTokenizer
from attack_sim.attacktutils import  create_gradient_leakage_client  # Custom client to simulate gradient leakage attacks
from client.clientutils import (
    create_flower_client # Further split and preprocess data into train/test sets
)
from server.serverutils import split_data  # Splits data among clients in IID or non-IID fashion
from flwr.client import start_client  # Starts a Flower client to participate in FL
from data.data_utils import load_dataset_hf, prepare_data, preprocess_and_split, load_config

def prepare_client_data(dataset_name, dataset_type, input_column, output_column, model_type, num_clients):
    """
    Prepare and split data for federated learning clients.

    Steps:
    1. **Load Tokenizer (if text dataset)**:
       - Initializes tokenizer if the dataset type is `text`.
    
    2. **Load Dataset**:
       - Uses `load_dataset_hf` to fetch dataset from local or Hugging Face datasets.
    
    3. **Prepare Data**:
       - Preprocesses the dataset (e.g., tokenization, normalization).
    
    4. **Train-Test Split**:
       - Splits the dataset into training and testing sets.
    
    5. **Distribute Data Among Clients**:
       - Splits training and test sets equally across clients in IID fashion.
    
    6. **Reshape for LSTM**:
       - If the model type is `lstm`, reshapes the data to the expected input dimensions.

    Returns:
    - `client_data`: Training data distributed among clients.
    - `test_data`: Testing data distributed among clients.
    - `input_shape`: Shape of the input features.
    - `num_classes`: Number of unique output classes.
    """
    # Load tokenizer if required
    tokenizer = None
    if dataset_type == "text":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    input_shape = X_train.shape[1]
    if(dataset_type=='traditional'):
        input_shape = X_train.shape[1:]
        print("input_s   hape",input_shape)  

    return client_data, test_data, input_shape, len(np.unique(Y_train))

def start_flower_client(client_id, input_shape, num_classes, model_type, client_data, test_data):
    """
    Initialize and start a Flower client with gradient leakage simulation.

    Steps:
    1. **Load Client's Data**:
       - Extracts training and testing data specific to the client.

    2. **Create Gradient Leakage Client**:
       - Initializes a custom client that simulates gradient leakage attacks during FL.

    3. **Start the Client**:
       - The client connects to the FL server and begins the training/evaluation process.

    Parameters:
    - `client_id`: Unique identifier for the client.
    - `input_shape`: Shape of the input features.
    - `num_classes`: Number of output classes.
    - `model_type`: Type of the model (e.g., dense, LSTM).
    - `client_data`: Training data for the client.
    - `test_data`: Testing data for the client.
    """
    X_client_train, Y_client_train = client_data[client_id]
    X_client_test, Y_client_test = test_data[client_id]

    # Create the model
    print(f"Initializing Client {client_id}")
   #  client = create_gradient_leakage_client(
   #      input_shape, num_classes, model_type, X_client_train, Y_client_train, X_client_test, Y_client_test
   #  )
    client = create_flower_client(
         input_shape, num_classes, model_type, X_client_train, Y_client_train, X_client_test, Y_client_test
      )


    # Start Flower client
    print(f"Starting Client {client_id}")
    start_client(server_address=f"127.0.0.2:8080", client=client)

def main(dataset_name, dataset_type="traditional", model_type="dense", input_column=None, output_column=None):
    """
    Main function to prepare data and start federated learning clients.

    Steps:
    1. **Define Number of Clients**:
       - Set the number of clients participating in FL.
    
    2. **Prepare Data for Clients**:
       - Loads dataset, preprocesses it, and splits data for each client.
    
    3. **Start Clients**:
       - Each client is initialized and starts participating in FL.

    Parameters:
    - `dataset_name`: Name of the dataset to be used.
    - `dataset_type`: Type of data (e.g., traditional, text).
    - `model_type`: Type of the model to be used (e.g., dense, LSTM).
    - `input_column`: Column name for the input features.
    - `output_column`: Column name for the target labels.
    """
    num_clients = 2  # Number of federated clients

    # Prepare data for clients
    client_data, test_data, input_shape, num_classes = prepare_client_data(
        dataset_name, dataset_type, input_column, output_column, model_type, num_clients
    )

    # Start a specific client (e.g., client 1)
    client_id = 2  # Example: For simplicity, starting one client
    start_flower_client(client_id, input_shape, num_classes, model_type, client_data, test_data)

if __name__ == "__main__":
    """
    Script Entry Point:
    1. **Load Configuration**:
       - Reads dataset and client configuration from `config.yaml`.
    
    2. **Run Main Function**:
       - Initializes the federated clients and starts the FL process.
    """
    # Load client configuration
    config = load_config("config.yaml")
    client1_config = config["client2"]

    # Run main
    main(
        dataset_name=client1_config["dataset_name"],
        dataset_type=client1_config["dataset_type"],
        model_type=client1_config["model_type"],
        input_column=client1_config["input_column"],
        output_column=client1_config["output_column"],
    )
