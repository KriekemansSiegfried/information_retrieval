import torch
import torch.nn as nn
import numpy as np


class BasicModel(nn.Module):

    def __init__(self, img_dim, txt_dim, hidden_dim, c):
        super().__init__()
        # instantiate basic image projection to shared feature space
        self.img_proj = torch.nn.Sequential(
            torch.nn.Linear(img_dim, hidden_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim * 2, hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, c),
        )

        # instantiate basic text projection to shared feature space
        self.txt_proj = torch.nn.Sequential(
            torch.nn.Linear(txt_dim, 256),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            torch.nn.Linear(256, c),
        )

    def forward(self, x, y):
        # project image to shared feature space
        F = self.img_proj(x).t()

        # project description to shared feature space
        G = self.txt_proj(y).t()

        B = torch.sign(F + G)

        # unique, counts = np.unique(B.data.numpy(), return_counts=True)
        # count_dict = dict(zip(unique, counts))
        # print('B 1: {}, B 0: {}'.format(count_dict[1], count_dict[-1]))
        #
        return F, G, B
