# input data path
input_data_path = "../input"
ouput_data_path = "../output"

# split the train, validation and test proportion (combine to be 1)
train_rate = 0.8
vali_rate = 0.1
test_rate = 0.1

# define validation stop rate
vali_stop = 0.898

# define size of batch (better if can be divide by 60000 and 2 to the nth)
batch_size = 32
# define batch normalization (boolean)
do_batch_norm = True

# define the dropout rate here [0,1]
dropout_rate = 0.1

# define the learning rate here ( > 0 )
learning_rate = 0.01

# define number of epoch
epoch = 100

# define the decay rate([0,1]) and regularizer (L1 or L2) here
regularizer = "L2"
decay_rate = 0.0005

# define way of doing optimization
optimization = "momentum"

# num of hidden layer features
num_hide1 = 96
num_hide2 = 64
num_hide3 = 32
