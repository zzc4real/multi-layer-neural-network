from optimizer import *


class standard(object):
    def __init__(self, avg_decay = 0.9, optimizer = None):
        self.m = None
        self.gama = None
        self.beta = None
        self.dgama = 0
        self.dbeta = 0
        self.z = None
        self.z_hat = None
        self.mean = None
        self.var = None
        self.std = None
        self.avg_decay = avg_decay
        self.optimizer = optimizer
        self.avg_mean = 0
        self.avg_var = 0

    def __init_param(self, din):
        self.gama = np.ones((din, 1))
        self.beta = np.zeros((din, 1))
        self.avg_mean = np.zeros((din, 1))
        self.avg_var = np.zeros((din, 1))

    def clone(self):
        opt = None
        if (self.optimizer is not None):
            opt = self.optimizer.clone()
        return standard(self.avg_decay, opt)

    def forward(self, z, training, epsilon = 1e-10):
        if (self.gama is None and self.beta is None):
            self.__init_param(z.shape[0])

        if not training:
            z_hat = (z - self.avg_mean) / np.sqrt(self.avg_var + epsilon)
            return (self.gama * z_hat) + self.beta

        self.z = z
        # calculate the mean of each features
        self.mean = np.mean(z, axis=1, keepdims=True)
        # variance
        self.var = np.var(z, axis=1, keepdims=True)
        # standard deviation
        self.std = np.sqrt(self.var + epsilon)
        # normalize
        self.z_hat = (self.z - self.mean) / self.std

        self.avg_mean = (self.avg_decay) * self.avg_mean + (1 - self.avg_decay) * (self.mean)
        self.avg_var = (self.avg_decay) * self.avg_var + (1 - self.avg_decay) * (self.var)

        # scale and shift
        return (self.gama * self.z_hat) + self.beta

    def backward(self, din_norm):
        # in_norm = gama * input_hat + beta
        # therefore, din_hat = din_norm * gama
        din_hat = din_norm * self.gama

        m = self.z.shape[1]
        # dgama = din_norm * input_hat
        # din_norm is (n_feature, m), input_hat is (n_feature, m)
        # where gama is (n_feature, 1), sum the product along axis 1
        self.dgama = np.sum(din_norm * self.z_hat, axis=1, keepdims=True) / m
        # the same applied to dbeta
        self.dbeta = np.sum(din_norm, axis=1, keepdims=True) / m

        # now comput din_norm/din

        # the following derivative can be found in paper,
        dvar = np.sum(din_hat * (self.z - self.mean), axis=1, keepdims=True) * ((self.std ** -3) / -2)

        dmean = np.sum(din_hat * (-1 / self.std), axis=1, keepdims=True) + dvar * np.sum(-2 * (self.z - self.mean),
                                                                                         axis=1, keepdims=True) / m

        din = din_hat / self.std + (dvar * (2 * (self.z - self.mean) / m)) + dmean / m

        return din

    def update(self, lr):
        if (self.optimizer is None):
            self.gama = self.gama - lr * self.dgama
            self.beta = self.beta - lr * self.dbeta
        else:
            self.gama = self.optimizer.update_W(lr, self.gama, self.dgama)
            self.beta = self.optimizer.update_b(lr, self.beta, self.dbeta)
