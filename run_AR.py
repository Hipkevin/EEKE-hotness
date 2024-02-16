import random
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from util.dataTool import getWordFrequency


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(data_loader, model):
    print('---Test---')
    model.eval()
    with torch.no_grad():
        Y_hat = []
        Y = []

        for item in data_loader:
            x, y = item[:, :-1].to(device), item[:, -1].to(device)
            y_hat = model(x)

            Y_hat.append(y_hat.squeeze())
            Y.append(y)

        Y_hat = torch.cat(Y_hat, 0)
        Y = torch.cat(Y, 0)
        MAE_criterion = torch.nn.L1Loss(reduction='mean')
        MAE = MAE_criterion(Y_hat, Y)
        MSE = criterion(Y_hat, Y)

        print(f"Test: MSE: {MSE}, MAE: {MAE}")


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.Dropout(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    epochs = 10
    lr = 1e-5
    weight_decay = 1e-5
    window_size = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(7)

    print('Data Loading...')
    train_set, test_set = getWordFrequency('data/KW_freq_matrix.csv', threshold=15, window_size=window_size)

    train_set = torch.tensor(train_set).float()
    test_set = torch.tensor(test_set).float()

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, pin_memory=True)

    model = MLP(window_size - 1).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=lr, weight_decay=weight_decay)

    print('---Train---')
    for epoch in range(1, epochs + 1):

        model.train()
        for idx, item in enumerate(train_loader):
            x, y = item[:, :-1].to(device), item[:, -1].to(device)

            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if idx % 10000 == 0:
                print(f'epoch: {epoch} batch: {idx} | loss: {loss}')

        evaluate(test_loader, model)