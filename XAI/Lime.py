import tensorflow as tf
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

print(tf.__version__)

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)

loadPath = "../SavedModels/BestModel/"
DirTestImg = "HIVImages/SmilesColorImages/Test/"
DirTensorArray = "HivData/SmilesColorArray/"

# Model config
batch_size = 32
nb_epoch = 200
verbose = 1

# change depending on image, 180 for mol images, 0 for others
rotation_range = 0

params = {
    'conv1_units': 16,
    'conv2_units': 16,
    'conv3_units': 16,
    'conv4_units': 16,
    'conv5_units': 16,
    'conv6_units': 16,
    'num_block1': 3,
    'num_block2': 3,
    'num_block3': 3,
    'dropval': 0.5,
}

# Load data to be explained
df_test = 

input_shape = X_test.shape[1:]

# Load pretrained model
filecp = loadPath + "_bestweights_trial_" + ".hdf5"
model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=2, lr=0.0001)
model.load_weights(filecp, by_name=True)

print(model.summary())