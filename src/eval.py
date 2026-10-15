import torch
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from config import *
from model import DiffusionModel

PLAY_ANIMATION = False

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

    X_series = model_t.generate()
    X_series = [x.cpu().detach().view(28, 28) for x in X_series]

    if PLAY_ANIMATION: 
        #Generate animation
        fig, ax = plt.subplots()
        ax.set_axis_off()
        im = ax.imshow(X_series[0])

        def update(i):
            im.set_data(X_series[i])
            return [im]

        ani = anime.FuncAnimation(
            fig, update,
            frames = len(X_series),
            interval= 1,
            blit=True
        )
    else:
        plt.imshow(X_series[-1])
    plt.show()
