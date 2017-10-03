import torch as torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from datasetloader import DatasetLoader
from train import Train
from models.combinedmodel import CombinedModel
from models.featuresmodel import FeaturesModel
from models.imagemodel import ImageModel
from models.postermodel import PosterModel
from torch.autograd import Variable


def getNetwork(model, file_name, create_new=True, verbose=True, shuffle=True):
    loader = DatasetLoader({
        'batch_size': 400,
        'num_workers': 8 if torch.cuda.is_available() else 0,
        'shuffle': shuffle
    })
    network = Train(model,
                    loader,
                    model_filename=file_name,
                    create_new=create_new,
                    print_every=1,
                    verbose=verbose)
    return network


def trainModel(network, start_lr, epochs, decay_steps, weight_decay=1e-3):
    loss_fn = nn.CrossEntropyLoss().type(network.data_type)
    for i in range(decay_steps):
        decay = 10 ** i
        lr = start_lr / decay
        network.train(loss_fn, optim.Adam(network.model.parameters(), lr=lr, weight_decay=weight_decay),
                      num_epochs=epochs)
    return network


def start():
    network1 = trainModel(getNetwork(CombinedModel, "model1-1.pt"), 1.092705e-02, 6, 4, 9.722207e-04)
    network2 = trainModel(getNetwork(CombinedModel, "model1-2.pt"), 1.092705e-02, 6, 4, 9.722207e-04)
    network3 = trainModel(getNetwork(FeaturesModel, "model2-1.pt"), 2.996677e-02, 6, 4, 1.523906e-04)
    network4 = trainModel(getNetwork(FeaturesModel, "model2-2.pt"), 2.996677e-02, 6, 4, 1.523906e-04)
    network5 = trainModel(getNetwork(FeaturesModel, "model2-3.pt"), 2.996677e-02, 6, 4, 1.523906e-04)
    network6 = trainModel(getNetwork(ImageModel, "model3-1.pt"), 2.996677e-02, 6, 4, 1.523906e-04)
    network7 = trainModel(getNetwork(ImageModel, "model3-2.pt"), 2.996677e-02, 6, 4, 1.523906e-04)
    network8 = trainModel(getNetwork(PosterModel, "model4-1.pt"), 5.993566e-04, 6, 4, 1.523906e-04)
    network9 = trainModel(getNetwork(PosterModel, "model4-2.pt"), 5.993566e-04, 6, 4, 1.523906e-04)
    pass


def gridSearch():
    lerning_rates = 10 ** np.random.uniform(-4, -1, 12)

    networks = [
        CombinedModel,
        FeaturesModel,
        ImageModel,
        PosterModel];

    for network in networks:
        best_acc = 0
        best_model = None
        for lr in lerning_rates:
            model = trainModel(getNetwork(network, "model-gs.pt", verbose=False, shuffle=False), lr, 2, 1,
                                   1.523906e-04)
            acc = model.check_val_accuracy()
            print('lr %e val accuracy: %f' % (lr, acc))
            if (best_acc < acc):
                best_acc = acc
                best_model = (lr, acc)
        print('----- Best Results -----')
        print('lr %e val accuracy: %f' % (best_model[0], best_model[1]))
        print(' ')
        print(' ')


def check_accuracy(loader, models, predictionFunction):
    num_correct = 0
    num_samples = 0
    for model in models:
        model.eval()
    for x, x1, y in loader:
        x_var = Variable(x.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),
                         volatile=True)
        x1_var = Variable(x1.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor),
                          volatile=True)
        preds = predictionFunction(x_var, x1_var, models)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc;


def probabilityPrediction(x, x1, models):
    result = None
    for model in models:
        scores = model(x, x1)
        probs = nn.Softmax()(scores)
        if (result is None):
            result = probs
        else:
            result += probs
    _, result = result.data.cpu().max(1)
    return result


def majorityPrediction(x, x1, models):
    result = None
    for model in models:
        scores = model(x, x1)
        _, preds = scores.data.cpu().max(1)
        if (result is None):
            result = preds
        else:
            result += preds
    result[result < math.ceil(len(models) / 2)] = 0
    result[result > 0] = 1
    return result


def checkAccAllModels():
    network1 = getNetwork(CombinedModel, "model1-1.pt", False)
    network2 = getNetwork(CombinedModel, "model1-2.pt", False)
    network3 = getNetwork(FeaturesModel, "model2-1.pt", False)
    network4 = getNetwork(FeaturesModel, "model2-2.pt", False)
    network5 = getNetwork(FeaturesModel, "model2-3.pt", False)
    network6 = getNetwork(ImageModel, "model3-1.pt", False)
    network7 = getNetwork(ImageModel, "model3-2.pt", False)
    network8 = getNetwork(PosterModel, "model4-1.pt", False)
    network9 = getNetwork(PosterModel, "model4-2.pt", False)

    models = [network1.model,
              network2.model,
              network3.model,
              network4.model,
              network5.model,
              network6.model,
              network7.model,
              network8.model,
              network9.model]

    loader = DatasetLoader({
        'batch_size': 235,
        'num_workers': 8 if torch.cuda.is_available() else 0
    })
    print('---------------start-----------------')
    network1.check_test_accuracy()
    network2.check_test_accuracy()
    network3.check_test_accuracy()
    network4.check_test_accuracy()
    network5.check_test_accuracy()
    network6.check_test_accuracy()
    network7.check_test_accuracy()
    network8.check_test_accuracy()
    network9.check_test_accuracy()
    print('Average Probability Accurancy')
    check_accuracy(loader.get_test_loader(), models, probabilityPrediction)
    print('Majority Prediction Accurancy')
    check_accuracy(loader.get_test_loader(), models, majorityPrediction)


start()
checkAccAllModels()
#gridSearch()
