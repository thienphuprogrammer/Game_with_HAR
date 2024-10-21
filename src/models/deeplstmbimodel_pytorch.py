import torch
import torch.nn as nn


class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Define the layers for the Residual Block
        # Define layer normalization for input size
        self.morm1 = nn.LayerNorm(input_size)
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, input_size // 2)

        # Layer normalization for input size//2
        self.morm2 = nn.LayerNorm(input_size // 2)
        # Second fully connected layer
        self.fc2 = nn.Linear(input_size // 2, output_size)

        # Final fully connected layer
        self.fc3 = nn.Linear(input_size, output_size)

        # Define the activation function
        self.activation = nn.ELU()

    def forward(self, x):
        out = self.activation(self.morm1(x))
        # Calculate the residual
        skip = self.fc3(x)

        # Apply layer normalization, fully connected layer, and activation function
        out = self.activation(self.morm2(self.fc1(out)))
        out = self.fc2(out)

        # Add the residual to the output
        return out + skip


class DeepLSTMBiModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            patch_size,
            lstm_layers,
            hidden_sizes,
            number_block,
            dropout=0.5,
    ):
        super(DeepLSTMBiModel, self).__init__()

        # Deep LSTM: Bidirectional with many layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_sizes[0] * 2)

        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[0] * 2,
            hidden_size=hidden_sizes[1],
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_sizes[1] * 2)

        self.lstm3 = nn.LSTM(
            input_size=hidden_sizes[1] * 2,
            hidden_size=hidden_sizes[2],
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_sizes[2] * 2)

        self.lstm4 = nn.LSTM(
            input_size=hidden_sizes[2] * 2,
            hidden_size=hidden_sizes[3],
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(hidden_sizes[3] * 2)

        # blocks = [
        #     ResBlockMLP(hidden_sizes, hidden_sizes // 8) for _ in range(number_block)
        # ]
        # self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_sizes[3] * 2, output_size)

    def initialize_hidden(self, batch_size, hidden_size):
        # Initialize hidden and cell states (2 for bidirectional, * layers)
        num_directions = 2
        hidden = torch.zeros(
            self.lstm.num_layers * num_directions, batch_size, hidden_size
        )
        cell = torch.zeros(
            self.lstm.num_layers * num_directions, batch_size, hidden_size
        )
        return hidden, cell

    def forward(self, input_data):
        # LSTM layer 1
        out, _ = self.lstm1(input_data)
        out = self.dropout1(out)
        out = self.norm1(out)

        # LSTM layer 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.norm2(out)

        # LSTM layer 3
        out, _ = self.lstm3(out)
        out = self.dropout3(out)
        out = self.norm3(out)

        # LSTM layer 4
        out, _ = self.lstm4(out)
        out = self.dropout4(out)
        out = self.norm4(out)

        # # Residual blocks
        # out = self.blocks(out)

        # Fully connected layer
        out = self.fc(out[:, -1, :])
        return out
