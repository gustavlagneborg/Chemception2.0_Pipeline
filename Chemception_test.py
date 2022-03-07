from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# load or create data with if statement
DirTrainImg = "Images/MolFromSmilesImages/Train/"
DirTestImg = "Images/MolFromSmilesImages/Test/"
DirTrainArray = "Data/MolFromSmilesArray/"
DirTestArray = "Data/MolFromSmilesArray/"

# Loading trainig data
if (os.path.exists("Data/MolFromSmilesArray/Train/X_train.pickle") and 
    os.path.exists("Data/MolFromSmilesArray/Train/y_train.pickle")):
    pickle_in_X = open("X_train.pickle","rb")
    X_train = pickle.load(pickle_in_X)

    pickle_in_y = open("y_train.pickle","rb")
    y_train = pickle.load(pickle_in_y)
else:
    print("Local train data was found" + "\n")
    X_train, y_train = tensorDataPrep(loadPath=DirTrainImg, savePath=DirTrainArray, testOrTrain="Train")

    # Loading test data
if (os.path.exists("Data/MolFromSmilesArray/Test/X_test.pickle") and 
    os.path.exists("Data/MolFromSmilesArray/Test/y_test.pickle")):
    pickle_in_X = open("X_test.pickle","rb")
    X_test = pickle.load(pickle_in_X)

    pickle_in_y = open("y_test.pickle","rb")
    y_test = pickle.load(pickle_in_y)
else:
    print("Local test data was found" + "\n")
    X_test, y_test = tensorDataPrep(loadPath=DirTestImg, savePath=DirTestArray, testOrTrain="Test")

print(X_train.shape)
print(y_train.shape)
    
# Check if GPU is available
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# Model setup
input_shape = X_train.shape[1:]
input_img = Input(shape=input_shape)

x = Inception0(input_img)
x = Inception(x)
x = Inception(x)
od=int(x.shape[1])
x = MaxPooling2D(pool_size=(od,od), strides=(1,1))(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
output = Dense(1, activation='linear')(x)
model = Model(inputs=input_img, outputs=output)
print(model.summary())