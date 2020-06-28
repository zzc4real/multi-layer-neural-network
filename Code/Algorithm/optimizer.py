import numpy as np


class Momentum(object):
    def __init__(self, momentum_term = 0.9):

        self.gama = 0.9
        self.v_W = 0
        self.v_b = 0
        self.t = 0
        self.name = 'momentum'

    def clone(self):
        return Momentum(self.gama)

    def update_W(self, lr, W, grad_W):
        self.v_W = (self.gama * self.v_W) - (lr * grad_W)
        return W + self.v_W

    def update_b(self, lr, b, grad_b):
        self.v_b =(self.gama * self.v_b) - (lr * grad_b)
        return b + self.v_b


class Nesterov(object):
    def __init__(self, momentum_term = 0.9):

        self.gama = 0.9
        self.v_W = 0
        self.v_b = 0
        self.t = 0
        self.name = 'nesterov'
        self.v_W_last = 0
        self.v_b_last = 0


    def clone(self):
        return Nesterov(self.gama)

    def get_last_v_W(self):
        return self.v_W_last

    def get_last_v_b(self):
        return self.v_b_last


    def update_W(self, lr, W, grad_W):
        self.v_W_last = np.copy(self.v_W)
        self.v_W = (self.gama * self.v_W) + (lr * grad_W)
        return W - self.v_W

    def update_b(self, lr, b, grad_b):
        self.v_b_last = np.copy(self.v_b)
        self.v_b =(self.gama * self.v_b) + (lr * grad_b)
        return b - self.v_b


class AdaGrad(object):

    def __init__(self):

        self.epsilon = 1e-6
        self.G_W = 0
        self.G_b = 0
        self.t = 0
        self.name = 'adagrad'

    def clone(self):
        return AdaGrad()


    def update_W(self, lr, W, grad_W):

        self.G_W += np.square(grad_W)
        return W - (lr/(np.sqrt(self.G_W + self.epsilon))) * grad_W

    def update_b(self, lr, b, grad_b):
        self.G_b += np.square(grad_b)
        return b - (lr/(np.sqrt(self.G_b + self.epsilon))) * grad_b


class AdaDelta(object):

    def __init__(self, decay = 0.9):
        self.epsilon = 1e-8
        self.E_g_W = 0
        self.E_g_b = 0
        self.E_delta_W = 0
        self.E_delta_b = 0
        self.decay = decay
        self.t = 0
        self.name = 'adadelta'

    def clone(self):
        return AdaDelta(self.decay)

    def update_W(self, lr, W, grad_W):

        self.E_g_W = (self.decay * (self.E_g_W)) + ((1 - self.decay) * np.square(grad_W))

        RMS_g = np.sqrt(self.E_g_W + self.epsilon)

        RMS_delta_W = np.sqrt(self.E_delta_W + self.epsilon)

        lr = (RMS_delta_W/RMS_g)

        delta_W = -lr * grad_W

        W += delta_W

        self.E_delta_W = (self.decay * (self.E_delta_W)) + ((1 - self.decay) * np.square(delta_W))
        return W

    def update_b(self, lr, b, grad_b):

        self.E_g_b = (self.decay * (self.E_g_b)) + ((1 - self.decay) * np.square(grad_b))

        RMS_g = np.sqrt(self.E_g_b + self.epsilon)

        RMS_delta_b = np.sqrt(self.E_delta_b + self.epsilon)

        lr = (RMS_delta_b/RMS_g)

        delta_b = -lr * grad_b

        b += delta_b

        self.E_delta_b = (self.decay * (self.E_delta_b)) + ((1 - self.decay) * np.square(delta_b))
        return b


class RMSProp(object):
    def __init__(self, decay = 0.9):
        self.epsilon = 1e-6
        self.E_g_W = 0
        self.E_g_b = 0
        self.decay = decay
        self.t = 0
        self.name = 'rmsprop'

    def clone(self):
        return RMSProp(self.decay)


    def update_W(self, lr, W, grad_W):

        self.E_g_W = (self.decay * (self.E_g_W)) + ((1 - self.decay) * np.square(grad_W))
        RMS = np.sqrt(self.E_g_W + self.epsilon)
        delta = (-lr/RMS)*grad_W
        return W + delta

    def update_b(self, lr, b, grad_b):

        self.E_g_b = (self.decay * (self.E_g_b)) + ((1 - self.decay) * np.square(grad_b))
        RMS = np.sqrt(self.E_g_b + self.epsilon)
        delta = (-lr/RMS)*grad_b
        return b + delta


class Adam(object):
    def __init__(self, beta1 = 0.9, beta2 = 0.999 ,bias_correction = True):
        self.epsilon = 1e-12

        self.beta1 = beta1
        self.beta2 = beta2
        self.vW = 0
        self.mW = 0
        self.name = 'adam'
        self.vb = 0
        self.mb = 0
        self.bias_correction = bias_correction
        if(self.bias_correction):
            self.t = 0
        else:
            self.t = 1

    def clone(self):
        return Adam(beta1 = self.beta1, beta2 = self.beta2, bias_correction = self.bias_correction)

    def update_W(self, lr, W, grad_W):

        if(self.bias_correction):
            self.t+=1
        self.mW = (self.beta1 * self.mW + (1 - self.beta1) * grad_W)
        self.vW = (self.beta2 * self.vW + (1 - self.beta2) * np.square(grad_W))
        m_hat = self.mW
        v_hat = self.vW

        if(self.bias_correction):
            m_hat = self.mW/(1. - (self.beta1 ** self.t))
            v_hat = self.vW/(1. - (self.beta2 ** self.t))

        return W - lr * (m_hat/np.sqrt(v_hat + self.epsilon))

    def update_b(self, lr, b, grad_b):


        self.vb = (self.beta2 * self.vb + (1 - self.beta2) * np.square(grad_b))
        self.mb = (self.beta1 * self.mb + (1 - self.beta1) * grad_b)
        m_hat = self.mb
        v_hat = self.vb

        if(self.bias_correction):
            m_hat = self.mb/(1. - (self.beta1 ** self.t))
            v_hat = self.vb/(1. - (self.beta2 ** self.t))

        return b - lr * (m_hat/np.sqrt(v_hat + self.epsilon))
