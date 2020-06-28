import numpy as np

class cross_entropy(object):

    def dz(self, y, y_hat):

        return y_hat - y


    def loss(self, y, y_hat):
        # get the batch size
        m = y.shape[1]

        loss = (-1/m) * np.sum(np.sum((y * np.log(y_hat))))

        return loss