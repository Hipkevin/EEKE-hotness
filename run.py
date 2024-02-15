import torch
import argparse
import inspect
import random
import os
import numpy as np
import util.model

from torch_geometric.data import Data
from util.opts import training_opts


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_classes(module):
    """
    获取模型名称及类名
    :param module: 模型定义文件目录
    :return: 映射字典
    """

    classes = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and module.__name__ in str(obj):
            classes[name] = obj

    return classes


def main():
    MODELS = get_classes(util.model)

    parser = argparse.ArgumentParser()
    training_opts(parser)

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)

    year = 5
    num_words = 5000

    emb_dim = args.word_emb_dim
    window_size = args.window_size
    epochs = args.epochs_num
    cv = args.cv
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据接口
    graph_list = []
    for i in range(year):
        graph = Data()

        # (words_num, emb_dim)
        graph.w_x = torch.randn(num_words + i * 100, emb_dim)
        graph.s_x = torch.randn(num_words + i * 100, emb_dim)

        # (words_num, window_size)
        graph.y_memory = torch.randint(0, 51, (num_words + i * 100, window_size)).float()
        # (words_num, 1)
        graph.y = torch.randint(0, 51, (num_words + i * 100, 1)).float()

        # (2, edge_num)
        graph.edge_index = torch.randint(0, num_words + i * 100, ((i + 1) * (num_words + i * 100), 2)).T
        # (edge_num, 1)
        graph.edge_weight = torch.randint(0, num_words + i * 100, ((i + 1) * (num_words + i * 100), 1)).float()

        graph_list.append(graph)

    model = MODELS[args.model](args).to(args.device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)

    # train
    model.train()
    for epoch in range(1, epochs + 1):
        for graph in graph_list[:-round(cv * year)]:
            graph = graph.to(args.device)

            y_hat = model(graph)
            loss = criterion(y_hat, graph.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'epoch: {epoch} | loss: {loss}')

    # test
    model.eval()
    with torch.no_grad():
        Y_hat = []
        Y = []
        for graph in graph_list[-round(cv * year):]:
            graph = graph.to(args.device)

            Y_hat.append(model(graph))
            Y.append(graph.y)

        Y_hat = torch.cat(Y_hat, 0)
        Y = torch.cat(Y, 0)
        MAE_criterion = torch.nn.L1Loss(reduction='mean')
        MAE = MAE_criterion(Y_hat, Y)
        MSE = criterion(Y_hat, Y)

        print(f"Test: MSE: {MSE}, MAE: {MAE}")


if __name__ == '__main__':
    main()