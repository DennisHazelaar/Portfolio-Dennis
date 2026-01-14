import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        filters,
        units1,
        units2,
        input_size=(32, 1, 28, 28),
        n_classes=10,
    ):
        super().__init__()

        in_channels = input_size[1]

        # --- Convolutions ---
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters, filters, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters, filters, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # --- Global pooling (NO shape guessing) ---
        self.agg = nn.AdaptiveAvgPool2d((1, 1))

        # --- Dense head ---
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, n_classes),
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        return self.dense(x)
