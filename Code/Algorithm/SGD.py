import numpy as np


class Batch(object):
    def __init__(self, X, Y):
        self.loss = []
        self.accuracy = []
        # x is (128, 60000)
        self.x_features = X.shape[0]
        # y is (10, 60000)
        self.y_classes = Y.shape[0]
        # number of sample data
        self.m = X.shape[1]
        # map y with x
        self.map = np.vstack((X, Y))

    def shuffle(self, boo = True):
        # shuffle the sample
        # shuffle the array along the first index
        m_t = np.copy(self.map.T)
        np.random.shuffle(m_t)
        self.map = np.copy(m_t.T)

    def get_x(self):
        # return 0:127 line of data
        return self.map[:self.x_features, :]

    def get_y(self):
        # return after 127 line of data
        return self.map[self.x_features:, :]

    def getLoss(self):
        return self.loss

    def getAccuracy(self):
        return self.accuracy

    def reset(self):
        self.loss = []
        self.accuracy = []

    def fit(self, model, size=None):
        # reset loss and accuracy array in current layer
        self.reset()

        # if batch size is not given, default batch size = total size
        if size == None:
            size = self.m

        self.shuffle()

        # get the number of batch, x and y
        batch_num = self.m//size
        sample = self.get_x()
        target = self.get_y()

        # start executing by each batch
        for i in range(batch_num):

            start = i * size
            end = start + size
            mini_X = sample[:, start:end]
            mini_Y = target[:, start:end]

            # data forward to next layer
            mini_Y_hat = model.forward(mini_X, training = True)

            self.loss.append(model.cost.loss(mini_Y, mini_Y_hat) + model.get_reg_loss())

            # compute accuracy for this mini batch
            self.accuracy.append(np.mean(np.equal(np.argmax(mini_Y, 0), np.argmax(mini_Y_hat, 0))))

            # compute gradient
            mini_dz = model.cost.dz(mini_Y, mini_Y_hat)

            # execute back-forward
            model.backward(mini_dz)

            model.update()