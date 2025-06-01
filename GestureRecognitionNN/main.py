import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset
from neural_network import *

# Importing dataset
try:
    with open(r'..\..\videos\landmarks_collection.pkl', 'rb') as f:
        dataset = pickle.load(f)
except Exception as e:
    print(f'Error reading pickle data: {e}\n '
          f'If "landmarks_collection.pkl" is missing, consider running "dataset_building.py".')

# Transform dataset from python list to numpy array
dataset = np.array(dataset)

# Split into features and labels
X = dataset[:,:63]
y = dataset[:,63:]

# Transform numpy arrays into pytorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).squeeze().long()
print(pd.Series(y).value_counts())

# Create train, validation and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, train_size=0.5)

# Create tensor dataset
train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)
test_dataset = TensorDataset(X_test, y_test)

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Create instance of the neural network
model = NeuralNetwork().to(DEVICE)

# Defining elements for the training
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Support lists for showing loss value
train_loss = []
valid_loss = []

# Train and validation loops
for ep in range(EPOCHS):
    train_loop(train_dataloader, model, loss_function, optimizer, train_loss)
    valid_loop(valid_dataloader, model, loss_function, valid_loss)

# Save model weights
torch.save(model.state_dict(), 'model_weights.pth')

# Lineplot and metrics for measuring quality of the model
accuracy, pred_list, y_list = test_loop(test_dataloader, model)
print(f'Accuracy: {accuracy}')

sns.set_theme()
sns.lineplot(x=np.arange(0,len(train_loss),1), y=train_loss, label='train')
sns.lineplot(x=np.arange(0,len(valid_loss),1), y=valid_loss, label='valid')
plt.legend()
plt.show()