import torch as torch
import torch.nn as nn
import torch.optim as optim
from datasetloader import DatasetLoader
from train import Train
from models.combinedmodel import CombinedModel
from models.featuresmodel import FeaturesModel
from models.imagemodel import ImageModel


def getNetwork1():
    loader = DatasetLoader({
        'batch_size': 409,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(CombinedModel,
                    loader,
                    model_filename="model1.pt",
                    create_new=True,
                    print_every=1)
    return network


def getNetwork2():
    loader = DatasetLoader({
        'batch_size': 409,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(FeaturesModel,
                    loader,
                    model_filename="model2.pt",
                    create_new=True,
                    print_every=1)
    return network


def getNetwork3():
    loader = DatasetLoader({
        'batch_size': 409,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(ImageModel,
                    loader,
                    model_filename="model3.pt",
                    create_new=True,
                    print_every=1)
    return network


def trainModel(network):
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-2, weight_decay=1e-3), num_epochs=10)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-3, weight_decay=1e-3), num_epochs=10)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-4, weight_decay=1e-3), num_epochs=10)
    return network


def start():
    network1 = trainModel(getNetwork1())
    network2 = trainModel(getNetwork2())
    network3 = trainModel(getNetwork3())


start()



