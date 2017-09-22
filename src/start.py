import torch as torch
import torch.nn as nn
import torch.optim as optim
from datasetloader import DatasetLoader
from train import Train
from models.combinedmodel import CombinedModel
from models.featuresmodel import FeaturesModel
from models.imagemodel import ImageModel
from models.postermodel import PosterModel
from torch.autograd import Variable


def getNetwork1(create_new=True):
    loader = DatasetLoader({
        'batch_size': 400,
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
        'batch_size': 400,
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
        'batch_size': 400,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(ImageModel,
                    loader,
                    model_filename="model3.pt",
                    create_new=create_new,
                    print_every=1)
    return network


def getNetwork4(create_new=True):
    loader = DatasetLoader({
        'batch_size': 400,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    network = Train(PosterModel,
                    loader,
                    model_filename="model4.pt",
                    create_new=create_new,
                    print_every=1)
    return network


def trainModel(network):
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-2, weight_decay=1e-3), num_epochs=8)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-3, weight_decay=1e-3), num_epochs=8)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-4, weight_decay=1e-3), num_epochs=8)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-5, weight_decay=1e-3), num_epochs=8)
    network.train(loss_fn, optim.Adam(network.model.parameters(), lr=1e-6, weight_decay=1e-3), num_epochs=8)
    return network


def start():
    network1 = trainModel(getNetwork1())
    network2 = trainModel(getNetwork2())
    network3 = trainModel(getNetwork3())
    network4 = trainModel(getNetwork4())
    pass


def check_accuracy(loader, model1, model2, model3, model4, predictionFunction):
    num_correct = 0
    num_samples = 0
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    for x, x1, y in loader:
        x_var = Variable(x.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),
                         volatile=True)
        x1_var = Variable(x1.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),
                          volatile=True)
        preds = predictionFunction(x_var, x1_var, model1, model2, model3, model4)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc;


def probabilityPrediction(x, x1, model1, model2, model3, model4):
    scores1 = model1(x, x1)
    scores2 = model2(x, x1)
    scores3 = model3(x, x1)
    scores4 = model4(x, x1)
    probs1 = nn.Softmax()(scores1)
    probs2 = nn.Softmax()(scores2)
    probs3 = nn.Softmax()(scores3)
    probs4 = nn.Softmax()(scores4)
    probs = probs1 + probs2 + probs3 + probs4;
    _, preds = probs.data.cpu().max(1)
    return preds


def majority(x, x1, model1, model2, model3, model4, value):
    scores1 = model1(x, x1)
    scores2 = model2(x, x1)
    scores3 = model3(x, x1)
    scores4 = model4(x, x1)
    _, preds1 = scores1.data.cpu().max(1)
    _, preds2 = scores2.data.cpu().max(1)
    _, preds3 = scores3.data.cpu().max(1)
    _, preds4 = scores4.data.cpu().max(1)
    preds = preds1 + preds2 + preds3 + preds4
    preds[preds < value] = 0
    preds[preds > 0] = 1
    return preds


def majority1(x, x1, model1, model2, model3, model4):
    return majority(x, x1, model1, model2, model3, model4, 1)


def majority2(x, x1, model1, model2, model3, model4):
    return majority(x, x1, model1, model2, model3, model4, 2)


def majority3(x, x1, model1, model2, model3, model4):
    return majority(x, x1, model1, model2, model3, model4, 3)


def checkAccAllModels():
    network1 = getNetwork1(False)
    network2 = getNetwork2(False)
    network3 = getNetwork3(False)
    network4 = getNetwork4(False)
    loader = DatasetLoader({
        'batch_size': 200,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    print('---------------start-----------------')
    network1.check_test_accuracy()
    network2.check_test_accuracy()
    network3.check_test_accuracy()
    network4.check_test_accuracy()
    print('Average Probability Accurancy')
    check_accuracy(loader.get_test_loader(), network1.model, network2.model, network3.model, network4.model,
                   probabilityPrediction)
    print('Majority1')
    check_accuracy(loader.get_test_loader(), network1.model, network2.model, network3.model, network4.model,
                   majority1)
    print('Majority2')
    check_accuracy(loader.get_test_loader(), network1.model, network2.model, network3.model, network4.model,
                   majority2)
    print('Majority3')
    check_accuracy(loader.get_test_loader(), network1.model, network2.model, network3.model, network4.model,
                   majority3)


start()
checkAccAllModels()
