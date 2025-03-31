import torch
import torch.nn as nn
from typing import Tuple


class MarioNet(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int) -> None:
        super(MarioNet, self).__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expected input height of 84, got {h}")
        if w != 84:
            raise ValueError(f"Expected input width of 84, got {w}")

        self.online = self._build_cnn(c, output_dim)
        self.target = self._build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, model: str = "online") -> torch.Tensor:
        """
        Forward pass through the specified network.
        """
      
        if model == "online":
            return self.online(x)
        elif model == "target":
            return self.target(x)
        else:
            raise ValueError("Model must be 'online' or 'target'.")

    def _build_cnn(self, c: int, output_dim: int) -> nn.Sequential:
        """
        Build the convolutional network.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
