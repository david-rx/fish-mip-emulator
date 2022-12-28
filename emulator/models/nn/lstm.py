from typing import Tuple
import torch
from torch import nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        torch.nn.init.constant_(self.fc.bias, 12) #Start with the mean roughly!

    def forward(self, x):
        # Set the initial hidden and cell states
        xt = x.transpose(0, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate the LSTM
        out, _ = self.lstm(xt, (h0, c0))


        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])
        return out

def train(train_data, train_labels, num_epochs):
    # Create an instance of the LSTM model
    input_size = 1  # number of features in the input data
    hidden_size = 32  # number of hidden units in the LSTM
    num_layers = 1  # number of LSTM layers
    num_classes = 1  # number of classes to predict (1 for regression)
    model = LSTM(input_size, hidden_size, num_layers, num_classes)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def make_lstm_features(features: np.array, labels: np.array) -> np.array:
    """
    The array is a 2d-array of shape (num_examples, num_features).
    Each row is a single example, and they are ordered by time.
    We want to convert it to (num_examples, num_time_steps, num_features)
    """
    num_examples = features.shape[0]
    sequences = []
    for i in range(20, num_examples):
        sequences.append(features[i - 20:i])
    sequences_arr = np.stack(sequences) # (num_examples, num_time_steps, num_features)
    return sequences_arr, labels[20:]

def make_lstm_features_3d(features: np.array, labels: np.array) -> Tuple[np.array, np.array]:
    """
    Input features is a 3d array of shape(num_time_steps, examples_per_timestamp, num_features)
    We want to convert it to (num_examples, num_time_steps, num_features)
    """
    num_examples = features.shape[0]
    sequences = []
    
    for i in range(20, num_examples):
        sequences.append(features[i-20:i])
    print(f"features shape before conversion: {features.shape}")
    #now we have a len(num_timesteps - 20) list of 3d arrays of shape (20, examples_per_timestamp, num_features)
    #we want to convert it to a 3d array of shape (num_examples, num_time_steps, num_features)
    #stacking would instead give a 4d array of shape (num_examples, 20, examples_per_timestamp, num_features)
    #first we will transpose the 3d arrays to shape (examples_per_timestamp, 20, num_features)
    #the we will concatenate them along the first axis to get a 3d array of shape (num_examples, 20, num_features)
    print("sequences[0] shape in list", sequences[0].shape)
    sequences_arr = np.concatenate([np.transpose(seq, (1, 0, 2)) for seq in sequences], axis=0)
    # sequences_arr = np.concatenate(sequences, axis=0)
    # labels_all = np.concatenate(labels, axis=0)
    print("sequences new shape", sequences_arr.shape)
    
    assert sequences_arr.shape == ((features.shape[0] -20) * features.shape[1], 20, features.shape[2])
    return sequences_arr, labels[20:].flatten()



def make_lstm_features_3d2(features: np.array):
    """
    Input array is a 3d array of shape(num_time_steps, examples_per_timestamp, num_features)
    We want to convert it to (num_examples, num_time_steps, num_features)
    Output shape should be (input_shape[1] * (input_shape[0] - 20), 20, input_shape[2])
    """
    pass