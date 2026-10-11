import torch
import matplotlib.pyplot as plt
import gzip
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from config import *
from model import DiffusionModel, alpha, beta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with gzip.open('data/train-images-idx3-ubyte.gz', 'r') as f:
    f.read(16)
    train_data = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'r') as f:
    f.read(8)
    train_label = f.read()

train_data = torch.tensor(list(train_data), device=device).view(60000, 784)
train_label = torch.tensor(list(train_label), device=device)

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'r') as f:
    f.read(16)
    test_data = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'r') as f:
    f.read(8)
    test_label = f.read()

test_data = torch.tensor(list(test_data), device=device).view(10000, 784)
test_label = torch.tensor(list(test_label), device=device)

def add_noise(raw_data, times):
    eps = torch.randn((raw_data.shape), device=device)
    a = alpha[times].unsqueeze(1)
    out = a.sqrt() * raw_data + (1-a).sqrt() * eps
    return out, eps

def get_specific_num(num):
    idx = []
    for i in range(60000):
        if train_label[i].item() == num:
            idx.append(i)
    data_num = train_data[idx, :].float().view(-1, 784)
    data_num = (data_num - data_num.mean(1, keepdim=True)) / data_num.std(1, keepdim=True)
    return data_num


def get_batch(images_set):
    epsilons = []
    idx = torch.randint(0, images_set.shape[0], (batch_size,), device=device)
    times = torch.randint(0, max_time, (batch_size,), device=device)
    Xt = images_set[idx, :]
    Xt, epsilons = add_noise(Xt, times)
    return Xt, times, epsilons

if __name__ == '__main__':
    
    for num in range(10):

        model = DiffusionModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        data = get_specific_num(num)

        lossi = []

        for i in tqdm(range(epochs)):
            Xt, times, epsilons = get_batch(data)
            ep_pred = model(Xt, times)
            loss = F.mse_loss(ep_pred, epsilons)

            lossi.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                optimizer.step()

        torch.save(model.state_dict(), f'model/model_num{num}.pth')