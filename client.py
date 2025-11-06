import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def train_local(model, dataloader, device, lr=1e-3, epochs=1, criterion=None,
                global_state=None, mu=0.0):
    """Train a model locally on one client and return weights, sample count, and loss."""
    model = deepcopy(model).to(device)
    model.train()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Move global weights to device if FedProx is used
    if global_state is not None:
        for k in global_state:
            global_state[k] = global_state[k].to(device)

    total_loss, total_samples = 0, 0

    for _ in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(imgs)
            loss = criterion(output, labels)

            # FedProx term: penalize deviation from global weights
            if global_state is not None and mu > 0:
                prox = 0.0
                for name, param in model.state_dict().items():
                    if param.is_floating_point():
                        prox += torch.norm(param - global_state[name]) ** 2
                loss += 0.5 * mu * prox

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    return model.state_dict(), total_samples, avg_loss


def evaluate_local(model, dataloader, device, criterion=None):
    """Evaluate model on local data and return average loss and accuracy."""
    model.eval()
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass
            output = model(imgs)
            loss = criterion(output, labels)
            total_loss += loss.item() * imgs.size(0)

            # Compute accuracy
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total
