import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """A simple CNN suitable for both classification and regression tasks."""
    
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Classification Functions

def train_classification(net, trainloader, optimizer, epochs, device: str):
    """Train the network for classification tasks."""
    criterion = torch.nn.CrossEntropyLoss()  # Classification loss
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test_classification(net, testloader, device: str):
    """Validate the network on classification tasks."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


# Regression Functions

def train_regression(net, trainloader, optimizer, epochs, device: str):
    """Train the network for regression tasks."""
    criterion = torch.nn.MSELoss()  # Regression loss (Mean Squared Error)
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device).float()  # Ensure targets are float
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def test_regression(net, testloader, device: str):
    """Validate the network on regression tasks."""
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, targets = data[0].to(device), data[1].to(device).float()  # Ensure targets are float
            outputs = net(images)
            total_loss += criterion(outputs, targets).item()
    return total_loss
