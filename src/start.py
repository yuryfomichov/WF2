import torch as torch
import torch.nn as nn
import torch.optim as optim
from datasetloader import DatasetLoader
from train import Train
from models.combinedmodel import CombinedModel
from models.featuresmodel import FeaturesModel
from models.imagemodel import ImageModel
from torch.autograd import Variable


def getNetwork1(create_new=True):
    loader = DatasetLoader({
        'batch_size': 409,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(CombinedModel,
                    loader,
                    model_filename="model1.pt",
                    create_new=create_new,
                    print_every=1)
    return network


def getNetwork2(create_new=True):
    loader = DatasetLoader({
        'batch_size': 294,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(FeaturesModel,
                    loader,
                    model_filename="model2.pt",
                    create_new=create_new,
                    print_every=1)
    return network


def getNetwork3(create_new=True):
    loader = DatasetLoader({
        'batch_size': 436,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(ImageModel,
                    loader,
                    model_filename="model3.pt",
                    create_new=create_new,
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

#start()
def check_accuracy(loader, model1, model2, model3):
    num_correct = 0
    num_samples = 0
    model1.eval()
    model2.eval()
    model3.eval()
    for x, x1, y in loader:
        x_var = Variable(x.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), volatile=True)
        x1_var = Variable(x1.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), volatile=True)
        scores1 = model1(x_var, x1_var)
        scores2 = model2(x_var, x1_var)
        scores3 = model3(x_var, x1_var)
        _, preds1 = scores1.data.cpu().max(1)
        _, preds2 = scores2.data.cpu().max(1)
        _, preds3 = scores3.data.cpu().max(1)
        preds = preds1 + preds2 + preds3;
        preds[preds > 1] = 1
        preds[preds <= 1] = 0
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc;


def checkAccAllModels():
    network1 = getNetwork1(False)
    network2 = getNetwork2(False)
    network3 = getNetwork3(False)
    loader = DatasetLoader({
        'batch_size': 10,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    check_accuracy(loader.get_test_loader(),network1.model, network2.model, network3.model)
checkAccAllModels()

