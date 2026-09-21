"""
feedforward.py -- the position-wise feed-forward network
"""

import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, config):
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)
        