import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  # Import Syft for Federated Learning

# Define the neural network model (simple example)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize a PySyft hook
hook = sy.TorchHook(torch)

# Create a virtual worker for the server
server = sy.VirtualWorker(hook, id="server")

# Create virtual workers for the clients
client1 = sy.VirtualWorker(hook, id="client1")
client2 = sy.VirtualWorker(hook, id="client2")

# Prepare the training data (for demonstration purposes, use the same data for all clients)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Split the data between the clients
client1_data, client2_data = torch.utils.data.random_split(train_dataset, [30000, 30000])

# Send data to clients
client1_dataset = sy.BaseDataset(client1_data, transform=transform).send(client1)
client2_dataset = sy.BaseDataset(client2_data, transform=transform).send(client2)

# Load data into DataLoader
client1_loader = torch.utils.data.DataLoader(client1_dataset, batch_size=32, shuffle=True)
client2_loader = torch.utils.data.DataLoader(client2_dataset, batch_size=32, shuffle=True)

# Initialize the global model
global_model = SimpleNN()

# Training function for each client
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    model.send(device)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    model.get()  # Get the model back to the server
    return model

# Federated learning loop
for epoch in range(5):  # Number of global epochs
    print(f"Epoch {epoch + 1}")
    
    # Distribute model to clients
    client_models = [global_model.copy().send(client) for client in [client1, client2]]
    optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]
    criterion = nn.CrossEntropyLoss()

    # Train the model on each client
    for client_model, client_loader, client_optimizer, client in zip(client_models, [client1_loader, client2_loader], optimizers, [client1, client2]):
        client_model = train(client_model, client_loader, client_optimizer, criterion, client)

    # Average the weights
    with torch.no_grad():
        global_state_dict = global_model.state_dict()
        for key in global_state_dict:
            global_state_dict[key] = torch.stack([client_model.state_dict()[key].float() for client_model in client_models]).mean(0)

    global_model.load_state_dict(global_state_dict)
    print("Updated global model")

print("Training complete.")
