import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Forward propagate RNN
        out, hidden = self.rnn(x, hidden)

        # Reshape output to (batch_size * sequence_length, hidden_size)
        out = out.contiguous().view(-1, self.hidden_size)

        # Decode hidden state of last time step
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state to zeros
        return torch.zeros(1, batch_size, self.hidden_size)
