'''

Conceptual File
create a learning model that DELIVER info to layer
info includes activation, initialization, batch norm,
              regularize, drop and optimization

'''
import numpy as np
import parameter
import matplotlib.pyplot as plt
from SGD import Batch
from cost import cross_entropy
from layer import hidden_layer
from initializer import Xavier, He
from activation import tanh, sigmoid, softmax, relu
import time

#     model = model = Model(train_X, train_Y, batch_size = get_batch_size() , drop = get_drop(),
#  regularizer = get_regularizer(), norm = get_norm(), optimizer = get_opt())
class model(object):
    def __init__(self,
                 train_data,
                 train_label,
                 batch_size=0,
                 learning_rate = 0.001,
                 drop = 0,
                 regularizer = None,
                 optimizer = None,
                 norm = None,
                 cost=cross_entropy()):

        # initialize layer
        self.layers = []

        # initialize optimize
        self.optimizer = optimizer

        # initialize regularizer
        self.drop = drop
        self.norm = norm
        self.regularizer = regularizer

        # initialize batches
        self.batch = Batch(train_data, train_label)
        self.batch_size = batch_size

        # initialize cost function
        self.cost = cost

        # initialize epoch (iteration times)
        self.epoch = 100

        # other attributes in class model
        # number of examples
        self.m = train_data.shape[1]
        # validation info
        self.cv_X = None
        self.cv_Y = None
        # how many different kinds of label
        self.classes = train_label.shape[0]
        # an array to records dimensions(num of neurons) of diff layers
        self.dims = [train_data.shape[0]]
        # learning rate
        self.lr = learning_rate

        # plot info
        self.printInfo = True
        self.printAt = 1
        self.Loss_plot = []
        self.Accu_plot = []

    def validation(self, vali_x, vali_y):
        self.cv_X = vali_x
        self.cv_Y = vali_y

    def add_layer(self,  n_out, ini = Xavier(), acti = relu(), drop = None):

        # drop update
        drop = self.drop

        # specify how many neurons are passing into current adding layer
        n_in = self.dims[-1]

        # create layer with 2 key params: # of output neuron and ini method
        layer = hidden_layer(n_in, n_out, ini)

        # set optimizer
        if (self.optimizer != None):
            layer.setOptimizer(self.optimizer.clone())

        # set dropout
        layer.setDropout(drop=drop)

        # set batch normalization
        if self.norm is not None:
            layer.setBatchNormalizer(self.norm.clone())

        # set activation function
        layer.setActivation(acti)

        # update model dimension array
        self.dims.append(n_out)
        # update model layers array
        self.layers.append(layer)
        print('creating layer with {} neurons, '.format(n_out), 'initialization: {}, '.format(ini.name), 'activation: {}'.format(acti.name))

    def fit(self, learning_rate=None, epoch=100):
        print("----------  START TRAINING  ----------")
        print("general information:")
        # update learning rate and epoch
        self.epoch = epoch
        if learning_rate != None:
            self.learning_rate = learning_rate
        print('optimization: {}'.format((self.optimizer.name)))
        print('learning rate: {}'.format((self.learning_rate)))
        print('dropout rate: {}'.format(self.drop))
        print('batch size: {}'.format((self.batch_size)))
        print('num of batch: {}'.format(self.m//self.batch_size))
        print('total epoch: {}'.format(self.epoch))


        # create loss and accuracy arrays
        total_loss_train = []
        total_loss_cv = []
        total_accu_train = []
        total_accu_cv = []

        # create timer
        strat_time = time.time()

        x_rem = epoch

        # start real learning
        for i in range(epoch):
            print('---------- executing {} / {} epoch '.format(i+1,epoch))
            self.batch.fit(self, size = self.batch_size)

            # get loss of all batch
            mean_loss_train = np.mean(self.batch.getLoss())
            mean_accu_train = np.mean(self.batch.getAccuracy())
            total_loss_train.append(mean_loss_train)
            total_accu_train.append(mean_accu_train)

            # cross validation
            cv_loss = 0
            cv_accu = 0

            if self.cv_X is not None and self.cv_Y is not None:
                pred_cv = self.predict(self.cv_X)
                cv_loss = self.cost.loss(self.cv_Y, pred_cv)
                # print('cv_loss: {}'.format(cv_loss))
                cv_accu = np.mean(np.equal(np.argmax(self.cv_Y, 0), np.argmax(pred_cv, 0)))

                total_loss_cv.append(cv_loss)
                total_accu_cv.append(cv_accu)

            self.Loss_plot.append(total_loss_train)
            self.Loss_plot.append(total_loss_cv)
            self.Accu_plot.append(total_accu_train)
            self.Accu_plot.append(total_accu_cv)

            if (self.printInfo and i % self.printAt == 0):
                print("train loss {:.5f}, train accur {:.3%}, val loss: {:.5f}, val accu: {:.3%}".format(mean_loss_train, mean_accu_train,cv_loss,cv_accu))

            if cv_accu > parameter.vali_stop:
                print("meet validation stop condition, stop training")
                x_rem = i+1
                break

        end_time = time.time()
        if (self.printInfo):
            s = end_time - strat_time
            print("Total training time {:.3f} s".format(s))

        return x_rem

    def forward(self, x, training = True):
        # set regularizer
        self.reset_regularizer()
        # layer forward
        for layer in self.layers:
            x = layer.forward(x, training=training, regularizer=self.regularizer)
        return x

    def get_reg_loss(self):
        if (self.regularizer is None):
            return 0
        else:
            return self.regularizer.get_loss(self.m)

    def add_last_layer(self,
                       ini=Xavier(),
                       acti=softmax()):
        n_in = self.dims[-1]
        n_out = self.classes
        layer = hidden_layer(n_in, n_out, ini, last_layer=True)
        # The activation function is softmax for the last layer
        layer.setActivation(softmax())

        # last layer dose not need Dropout
        layer.setDropout(drop=0)
        if (self.optimizer != None):
            layer.setOptimizer(self.optimizer.clone())

        self.layers.append(layer)
        print('LAST LAYER with initialization: {}, '.format(ini.name), 'activation: {}'.format(acti.name))

    def backward(self, dz):
        da = dz
        for layer in reversed(self.layers):
            da = layer.backward(da, self.regularizer)

    def update(self):
        for layer in self.layers:
            layer.update(self.lr)

    def reset_regularizer(self):
        if (self.regularizer is not None):
            self.regularizer.reset()

    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x, training=False)
        return x

    def test(self, test_x, test_y):
        print("----------  start testing  ----------")
        if (test_x is None):
            return
        pred_test = self.predict(test_x)
        test_accu = np.mean(np.equal(np.argmax(test_y, 0), np.argmax(pred_test, 0)))
        print(" *** Test accuracy: {:.2%} ***".format(test_accu))

    def plot(self, x_rem, accu=True, loss=True):
        print("----------  drawing learning result  ----------")
        if (accu or loss):

            # get x axis
            x = np.arange(x_rem + 1)
            # exclude epoch 0
            x = x[1:]

            # get subplot number
            subplot_number = 1
            i = 1
            if (accu and loss):
                subplot_number += 1

            # create figure
            plt.figure(1)

            if (accu):
                plt.subplot(subplot_number, 1, i)
                plt.plot(x, self.Accu_plot[0], label="train_accu")
                if (self.cv_X is not None):
                    plt.plot(x, self.Accu_plot[1], label="cv_accu")

                plt.xlabel("epoch")
                plt.ylabel("accu")

                plt.legend()

                i += 1

            if (loss):
                plt.subplot(subplot_number, 1, i)
                plt.plot(x, self.Loss_plot[0], label="train_loss")
                if (self.cv_X is not None):
                    plt.plot(x, self.Loss_plot[1], label="cv_loss")

                plt.xlabel("epoch")
                plt.ylabel("loss")

                plt.legend()

            plt.show()
