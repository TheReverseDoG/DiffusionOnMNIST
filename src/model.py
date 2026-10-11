import torch
import matplotlib.pyplot as plt
import gzip
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from config import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beta = torch.linspace(1e-4, 0.01, max_time, device=device)
alpha = torch.cumprod(1 - beta, dim=0) 

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding_table = nn.Embedding(max_time, t_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(pixel_size + t_emb_dim, hidden_num, bias=False),
            nn.LayerNorm(hidden_num), nn.SiLU(),
            nn.Linear(hidden_num, hidden_num, bias=False),
            nn.LayerNorm(hidden_num), nn.SiLU(),
            nn.Linear(hidden_num, hidden_num, bias=False),
            nn.LayerNorm(hidden_num), nn.SiLU(),
            nn.Linear(hidden_num, pixel_size)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding_table(t)
        out = torch.cat((t_emb, x), 1)
        out = self.net(out)
        return out

    @torch.no_grad()
    def generate(self, time = max_time):

        X_current = torch.randn((1, 784), device=device)
        X_last = torch.zeros((1, 784), device=device)
        T = time - 1
        for i in range(time):
            t = torch.tensor(T, device=device).view(1)
            epsilon = self.forward(X_current, t)

            if T == 0:
                break

            mu = (1 / (1 - beta[T].item()) ** 0.5) * (X_current - (beta[T].item() / (1 - alpha[t]) ** 0.5) * epsilon)
            std = beta[T].item() ** 0.5
            X_last = mu + std * torch.randn((1, 784), device=device)

            X_current = X_last
            T -= 1
        return X_last