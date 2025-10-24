import torch.nn as nn
import torch


class MLP(nn.Module):
    """ Simple MLP """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Uso la arquitectura que dan en la consigna
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    def forward(self, x):
        return self.net(x)
    
class CNN(nn.Module):
    def __init__(self, in_channels: int, hw: tuple, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),  # 10x10 -> 8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),           # 8x8 -> 6x6
            nn.ReLU(),
            nn.Flatten()
        )
        # calcular tamaño de la parte densa en tiempo de construcción
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, hw[0], hw[1])
            n_flat = self.features(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
