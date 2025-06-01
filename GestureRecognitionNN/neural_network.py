import torch
from torch import nn

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
DEVICE = "cpu"
print(f"Using {DEVICE} device.\n")

# Hyperparameters
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.01

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_function, optimizer, loss_value):
    model.train()
    train_loss_value = 0

    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_function(pred, y)
        train_loss_value += loss_function(pred, y).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss_value.append(train_loss_value/len(dataloader))


def valid_loop(dataloader, model, loss_function, loss_value):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss_value, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss_value += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss_value.append(valid_loss_value/num_batches)
    correct /= size

def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0
    pred_list = []
    y_list = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            pred_list.append(pred)
            y_list.append(y)

    return [correct / size, pred_list, y_list]