import numpy as np

class He(object):

    def __init__(self):
        self.name ='He'

    def get_W(self, n_in, n_out):
        W = np.random.uniform(
                low = -np.sqrt(6. / (n_in)),
                high = np.sqrt(6. / (n_in)),
                size = (n_out, n_in)
        )
        return W

    def get_b(self, n_out):
        return np.zeros((n_out, 1))


class Xavier(object):

    def __init__(self):
        self.name = 'Xavier'

    # Xavier initialization: kaiming uniform (-sqrt(6/n), sqrt(6/n)
    def get_W(self, n_in, n_out):
        W = np.random.uniform(
                low = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_out, n_in)
        )
        return W

    def get_b(self, n_out):
        return np.zeros((n_out, 1))
