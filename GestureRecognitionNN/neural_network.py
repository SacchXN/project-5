import torch
from torch import nn

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
DEVICE = "cpu"
print(f"Using {DEVICE} device.\n")

# Hyperparameters
EPOCHS = 150
BATCH_SIZE = 8
LEARNING_RATE = 0.01

# When working with 2 labels, the following setup works well
# l1: 63-16, l2: 16-4, l3: 4-3

# When working with 3 labels, probably the 4 output neurons in l2 are not enough to carry the necessary information
# to represent 3 classes, both training and validation loss hardly decrease. (Dataset is balanced, 0:875, 1:712, 2:687)
# Layers changed to l1: 63-16, l2: 16-8, l3: 8-3

# 16/06/25: It probably was the 'v' sign being hard to recognize. When 'fist' is added, the model loss starts decreasing
#           way faster. Epochs could be reduced from 150 to 70~100.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
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