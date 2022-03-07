from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

import keras
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# _____________________load or create data with if statement _____________________
DirTrainImg = "Images/MolFromSmilesImages/Train/"
DirTestImg = "Images/MolFromSmilesImages/Test/"
DirTrainArray = "Data/MolFromSmilesArray/"
DirTestArray = "Data/MolFromSmilesArray/"

# Loading trainig data
if (os.path.exists("Data/MolFromSmilesArray/trainData.pickle")):
    print("Local train data was found" + "\n")
    pickle_in = open("Data/MolFromSmilesArray/trainData.pickle","rb")
    trainData = pickle.load(pickle_in)

else:
    trainData = tensorDataPrep(loadPath=DirTrainImg, savePath=DirTrainArray, testOrTrain="Train")

    # Loading individual test data
if (os.path.exists("Data/MolFromSmilesArray/testData.pickle")):
    print("Local test data was found" + "\n")
    pickle_in = open("Data/MolFromSmilesArray/testData.pickle","rb")
    testData = pickle.load(pickle_in)
else:
    testData = tensorDataPrep(loadPath=DirTestImg, savePath=DirTestArray, testOrTrain="Test")

X_train = np.array(list(trainData.iloc[:, 0].values))
y_train = trainData.iloc[:, 1].values

X_test = np.array(list(testData.iloc[:, 0].values))
y_test = testData.iloc[:, 1].values

#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#  _____________________Model setup and 5-fold CV_____________________

input_shape = X_train.shape[1:]
kf = StratifiedKFold(5, shuffle=True, random_state=42) 
oos_y = []
oos_pred = []
fold = 0

for train, test in kf.split(X_train, y_train):
    fold+=1
    print(f"Fold #{fold}")
    
    modelName = "Chemceptoin_test_" + str(fold)

    X_train_cv = np.asarray(X_train[train])
    y_train_cv = np.asarray(y_train[train]).reshape(-1,1)
    X_test_cv = np.asarray(X_train[test])
    y_test_cv = np.asarray(y_train[test]).reshape(-1,1)
    
    input_img = Input(shape=input_shape)
    x = Inception0(input_img)
    x = Inception(x)
    x = Inception(x)
    od=int(x.shape[1])
    x = MaxPooling2D(pool_size=(od,od), strides=(1,1))(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_img, outputs=output, name=modelName)
    print(model.summary())
    break

#_____________________evaluation_____________________
# train/validation data

# test data
