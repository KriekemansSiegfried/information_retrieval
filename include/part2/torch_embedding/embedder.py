import torch.nn as nn
import torch.nn.functional as F
import torch


class Embedder(nn.Module):
    hidden_size = 1024

    def __init__(self, input_size, output_size=32):
        super(Embedder, self).__init__()

        self.input = nn.Linear(input_size, self.hidden_size * 2)
        self.hidden1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = self.output(x)
        return x
