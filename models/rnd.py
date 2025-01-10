import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetNetwork(nn.Module):
    """
    Sieć docelowa: losowo zainicjalizowane wagi, brak trenowania.
    """
    def __init__(self, input_channels=1, feature_dim=128):
        super(TargetNetwork, self).__init__()
        # Przykładowa architektura wzorowana na "Nature CNN" z artykułu Mnih et al.
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(7 * 7 * 64, feature_dim)  # 7x7, bo 84x84 -> [po konwolucjach]

        # Inicjalizacja wagi – *losowe* i zapamiętane
        # torch.nn.init.(...)

    def forward(self, x):
        # Zakładamy, że x ma kształt [batch_size, 1, 84, 84] lub [batch_size, 4, 84, 84] - w zależności od strategii
        # W artykule RND do sieci docelowej używa się 1 klatki (bez stakowania).
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PredictorNetwork(nn.Module):
    """
    Sieć predyktora: uczymy ją przewidywać to, co zwraca TargetNetwork.
    """
    def __init__(self, input_channels=1, feature_dim=128):
        super(PredictorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(7 * 7 * 64, feature_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
