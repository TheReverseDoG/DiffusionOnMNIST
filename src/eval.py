import torch
import matplotlib.pyplot as plt
from config import *
from model import DiffusionModel

while True:
    num = input("Please enter the number you want to generate(0 - 9), enter q to quit:")
    
    if num == 'q' or num == 'Q':
        break

    num = int(num)
    if num < 0 or num > 9:
        print("Invalid input! Please try again")
        continue

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_t = DiffusionModel().to(device)
    model_t.load_state_dict(torch.load(f'model/model_num{num}.pth'))

    model_t.eval()

    X_last = model_t.generate()

    plt.imshow(X_last.view(28, 28).cpu().detach())
    plt.show()
