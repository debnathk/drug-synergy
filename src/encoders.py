import torch.nn as nn

class OmicsEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        ) 

    def forward(self, x):
        return self.net(x)
    