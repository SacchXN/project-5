import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.01

# TO DO:
# Add train and test loop, loss function, optimizer, dataloader
