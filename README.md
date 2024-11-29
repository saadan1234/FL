# FedFusion: Federated Learning Simplified

FedFusion is an open-source project designed to simplify the process of implementing and experimenting with Federated Learning (FL) architectures. It provides a flexible framework for researchers and practitioners to explore FL techniques while maintaining scalability and ease of use.

This repository includes a basic implementation of a federated learning system with one server and multiple clients, facilitating experimentation in distributed machine learning.

---

## Features

- **Multi-client Support**: The system supports multiple clients for simulating realistic FL scenarios.
- **Flexible Design**: Easily extendable for custom datasets, models, and algorithms.
- **Federated Averaging**: Implements the FedAdam algorithm for distributed training.
- **Simple Setup**: Run the server and clients on separate devices or terminals with minimal configuration.

---

## Prerequisites

Ensure you have the following installed:

- Required dependencies (install using the instructions below)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saadan1234/FedFusion.git
   cd FedFusion
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

---

## Usage Instructions

### Overview

FedFusion requires three terminals or devices to simulate the federated learning environment:

1. **Server**: Central coordinator for the federated system.
2. **Client 1**: Simulates one participant in the federated learning setup.
3. **Client 2**: Simulates another participant in the federated learning setup.

Each component must be run simultaneously to facilitate proper communication.

### Steps

#### 1. Start the Server

In the first terminal (or device), navigate to the project directory and start the server:
```bash
python server.py
```

#### 2. Start Client 1

In the second terminal (or device), navigate to the project directory and start Client 1:
```bash
python client1.py 
```

#### 3. Start Client 2

In the third terminal (or device), navigate to the project directory and start Client 2:
```bash
python client2.py 
```

---

## How It Works

1. **Initialization**:
   - The server initializes a global model and waits for clients to connect.
   - Clients connect to the server and receive the initial global model.

2. **Training**:
   - Each client trains the model locally on its private dataset.
   - After local training, the clients send their model updates to the server.

3. **Aggregation**:
   - The server aggregates the updates using the Federated Averaging or Fed Adam algorithm whichever you may select.
   - The updated global model is sent back to the clients.

4. **Iteration**:
   - Steps 2 and 3 repeat for a defined number of rounds.

---

## Example Dataset

By default, FedFusion uses the Cifar10 dataset for training. You can replace it with your dataset by modifying the `task.py` script.

---

## Customization

To customize the architecture, modify the following files:

- **Task**: Edit `task.py` to change the neural network structure for your custom task.
- **Data Loader**: Edit `task.py` to include your dataset of your choice from any dataset available on HuggingFace.
- **Training Logic**: Edit `client.py` to customize the training procedure.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For questions or support, please contact [(https://www.linkedin.com/in/muhammad-saadan-975474249/)].

Happy Coding with FedFusion! 🎉
