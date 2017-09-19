import torch as torch
import torch.nn as nn
import torch.optim as optim
from .datasetloader import DatasetLoader
from .train import Train


def run():
    loader = DatasetLoader({
        'batch_size': 242,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(loader,
                    model_filename="model.pt",
                    create_new=False,
                    print_every=1)
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-5, weight_decay=1e-4), num_epochs=5)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-6, weight_decay=1e-4), num_epochs=5)


run()
