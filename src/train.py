import torch as torch
import time
from torch.autograd import Variable


class Train(object):
    def __init__(self, Model, loader, model_filename='model.pt', create_new=False, print_every=10):
        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.model_filename = model_filename
        self.create_new = create_new
        self.print_every = print_every
        self.loader = loader
        self.Model = Model
        self.best_acc = 0
        self.init()

    def init(self):
        self.init_model()
        if self.create_new:
            print('New model was created')
        else:
            try:
                self.load_model()
                print('Model was loaded from file')
            except:
                print('No model had been found. New model was created')

    def init_model(self):
        self.model = self.Model()
        self.model = self.model.type(self.data_type)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_filename)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_filename))

    def train(self, loss_fn, optimizer, num_epochs=1):
        for epoch in range(num_epochs):
            print('')
            print('--------------------------------------------------------------------------------------------------')
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))

            tic = time.time()
            read_data_tic = time.time()
            read_data_time = 0;
            forward_time = 0;
            convert_to_CUDA_time = 0;
            backward_time = 0;
            self.model.train()
            for t, (x, x1, y) in enumerate(self.loader.get_train_loader()):
                read_data_time += (time.time() - read_data_tic);

                if torch.cuda.is_available():
                    convert_to_CUDA_tic = time.time()
                    x, x1, y = x.cuda(async=True), x1.cuda(async=True), y.cuda(async=True)
                    convert_to_CUDA_time += (time.time() - convert_to_CUDA_tic);

                x_var = Variable(x, requires_grad=False)
                x1_var = Variable(x1, requires_grad=False)
                y_var = Variable(y.long())

                forward_time_tic = time.time()
                scores = self.model(x_var, x1_var)
                loss = loss_fn(scores, y_var)
                forward_time += (time.time() - forward_time_tic);

                if (t + 1) % self.print_every == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

                backward_time_tic = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_time += (time.time() - backward_time_tic);
                read_data_tic = time.time()
            print('Epoch done in t={:0.1f}s'.format(time.time() - tic))
            print('Reading data time t={:0.1f}s'.format(read_data_time))
            print('Convert to CUDA t={:0.1f}s'.format(convert_to_CUDA_time))
            print('Forward time t={:0.1f}s'.format(forward_time))
            print('Backward time t={:0.1f}s'.format(backward_time))

            acc = self.check_val_accuracy()
            if (acc >= self.best_acc):
                self.best_acc = acc;
                self.save_model();
        self.check_train_accuracy()
        self.check_test_accuracy()

    def check_train_accuracy(self):
        print('Checking accuracy on TRAIN set')
        return self.check_accuracy(self.loader.get_train_loader(False))

    def check_val_accuracy(self):
        print('Checking accuracy on VALIDATION set')
        return self.check_accuracy(self.loader.get_val_loader())

    def check_test_accuracy(self):
        print('Checking accuracy on TEST set')
        return self.check_accuracy(self.loader.get_test_loader())

    def check_accuracy(self, loader, stop_on=-1):
        num_correct = 0
        num_samples = 0
        self.model.eval()
        for x, x1, y in loader:
            x_var = Variable(x.type(self.data_type), volatile=True)
            x1_var = Variable(x1.type(self.data_type), volatile=True)
            scores = self.model(x_var, x1_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            if (stop_on >= 0 and num_samples > stop_on):
                break;

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc;
