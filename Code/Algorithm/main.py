import sys
import preprocessed
import parameter
import numpy as np
import h5py
from optimizer import Momentum, Nesterov, AdaGrad, AdaDelta, RMSProp, Adam
from model import model
from regularizer import L1, L2
from batchnorm import standard
from initializer import Xavier, He
from activation import tanh, sigmoid, softmax, relu

def get_batch_size():
    return parameter.batch_size


def get_dropout_rate():
    return parameter.dropout_rate


def get_lr():
    return parameter.learning_rate


def get_regularizer():
    decay_rate = parameter.decay_rate
    if (parameter.regularizer == "L2"):
        return L2(decay_rate)
    elif (config.Regularizer == "L1"):
        return L1(decay_rate)
    else:
        return None

def get_norm():
    if parameter.do_batch_norm:
        return standard()
    else:
        return None


def get_opt():
    opt = parameter.optimization.lower()
    if (opt == "adam"):
        return Adam()
    elif (opt == "adadelta"):
        return Adadelta()
    elif (opt == "adagrad"):
        return AdaGrad()
    elif (opt == "rmsprop"):
        return RMSProp()
    elif (opt == "nesterov"):
        return Nesterov()
    elif (opt == "momentum"):
        return Momentum()
    else:
        return None


def main(argv):

    # load and pre-process the data
    X, Predict_data, Y = preprocessed.data_preprocess(parameter.input_data_path)
    print('| Total train data | structure: {}'.format(X.shape))
    print('| Train Data label | structure: {}'.format(Y.shape))
    print('| Total test Data  | structure: {}'.format(Predict_data.shape))

    # split data into train, validation and test
    train_x, train_y, vali_x, vali_y, test_x, test_y = preprocessed.train_vali_test_split(X, Y, parameter.train_rate, parameter.vali_rate, parameter.test_rate)
    print("_______________________________________")
    print('after split\ntrain data shape:\t{}'.format(train_x.shape))
    print('train data label:\t{}'.format(train_y.shape))
    if vali_x is None:
        print(" after data pre-process, validation is none")
    else:
        print('validation data shape:\t{}'.format(vali_x.shape))
    if test_x is None:
        print(" after data pre-process, test data is none")
    else:
        print('test data shape:\t{}'.format(test_x.shape))
    print("_______________________________________")

    # create learning model
    # a model considers batch size, batch normalization, dropout rate, weight decay and way of optimization
    learn_model = model(train_x, train_y, batch_size=get_batch_size(), drop=get_dropout_rate(), learning_rate=get_lr(),
                        regularizer=get_regularizer(), norm=get_norm(), optimizer=get_opt())
    # set validation data into model
    learn_model.validation(vali_x, vali_y)

    # create neural layer1
    learn_model.add_layer(parameter.num_hide1, ini=He(), acti=relu())
    # layer2
    learn_model.add_layer(parameter.num_hide2, ini=He(), acti=relu())
    # layer3
    learn_model.add_layer(parameter.num_hide3, ini=He(), acti=relu())
    # layer4
    learn_model.add_last_layer(ini=Xavier(), acti=softmax())

    # start training
    x_rem = learn_model.fit(epoch=parameter.epoch, learning_rate=parameter.learning_rate)

    # start testing
    learn_model.test(test_x, test_y)

    # plot result
    learn_model.plot(x_rem, True, True)

    # start predict
    print("----------  finish predict, save to predict.h5  ----------")
    predict = learn_model.predict(x=Predict_data).T
    predict = np.argmax(predict, axis=1)
    # print(predict)

    f = h5py.File(parameter.ouput_data_path + "/Predicted_labels.h5", 'a')
    f.create_dataset('/predict', data=predict, dtype=np.float32)
    f.close()

if __name__ == "__main__":
    main(sys.argv)
