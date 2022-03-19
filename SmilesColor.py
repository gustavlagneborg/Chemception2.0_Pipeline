import statistics

import pandas as pd
from numpy import std
from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

import keras
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
import sys

print(tf.__version__)

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)

path = "SavedModels/SmilesColor/"
modelName = "T2_SmilesColorModel"
batch_size = 32
nb_epoch = 100
verbose = 1

# change depending on image, 180 for mol images, 0 for others
rotation_range = 0

params = {
    'conv1_units': 64,
    'conv2_units': 64,
    'conv3_units': 64,
    'conv4_units': 64,
    'conv5_units': 64,
    'conv6_units': 64,
    'num_block1': 1,
    'num_block2': 1,
    'num_block3': 1,
    'dropval': 0.5,
}

# _____________________load or create HivData with if statement _____________________
DirTrainImg = "HIVImages/SmilesColorImages/Train/"
DirTestImg = "HIVImages/SmilesColorImages/Test/"
DirTensorArray = "HivData/SmilesColorArray/"

# Loading trainig HivData
if os.path.exists("HivData/SmilesColorArray/X_Train.pickle"):
    print("Local train HivData was found" + "\n")
    pickle_in = open("HivData/SmilesColorArray/X_train.pickle", "rb")
    X_train_and_valid = pickle.load(pickle_in)

    pickle_in = open("HivData/SmilesColorArray/y_Train.pickle", "rb")
    y_train_and_valid = pickle.load(pickle_in)

else:
    print("Producing train HivData!" + "\n")
    X_train_and_valid, y_train_and_valid = tensorDataPrep(loadPath=DirTrainImg, savePath=DirTensorArray,
                                                          testOrTrain="Train")
    print("Done!")

    # Loading individual test HivData
if os.path.exists("HivData/SmilesColorArray/X_Test.pickle"):
    print("Local test HivData was found" + "\n")
    pickle_in = open("HivData/SmilesColorArray/X_Test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("HivData/SmilesColorArray/y_Test.pickle", "rb")
    y_test = pickle.load(pickle_in)

else:
    print("Producing test HivData!" + "\n")
    X_test, y_test = tensorDataPrep(loadPath=DirTestImg, savePath=DirTensorArray, testOrTrain="Test")
    print("Done!")

# test and valid split
X_train, X_valid, y_train, y_valid = train_test_split(
                                        X_train_and_valid,
                                        y_train_and_valid,
                                        test_size=0.3,
                                        random_state=random_state,
                                        shuffle=True,
                                        stratify=y_train_and_valid)

# print HivData shapes before oversampling
print("HivData shapes before oversampling: ")
print("X_train HivData shape: " + str(X_train.shape))
print("y_train HivData shape: " + str(y_train.shape) + "\n")

print("X_validation HivData shape: " + str(X_valid.shape))
print("y_validation HivData shape: " + str(y_valid.shape) + "\n")

# oversampling after split to ensure no sample leakage
balanced_indices = cs_data_balance(y_train.flatten().tolist())
X_train = X_train[balanced_indices]
y_train = y_train[balanced_indices]

balanced_indices = cs_data_balance(y_valid.flatten().tolist())
X_valid = X_valid[balanced_indices]
y_valid = y_valid[balanced_indices]

y_train = tf.one_hot(y_train.flatten(), depth=2)
y_train = tf.cast(y_train, tf.int32)
y_valid = tf.one_hot(y_valid.flatten(), depth=2)
y_valid = tf.cast(y_valid, tf.int32)
input_shape = X_train.shape[1:]

# show sample of a molecule
v = X_train_and_valid[0]
plt.imshow(v[:,:,:3])
plt.show()

# print HivData shapes after oversampling
print("HivData shapes after oversampling: ")
print("X_train HivData shape: " + str(X_train.shape))
print("y_train HivData shape: " + str(y_train.shape) + "\n")

print("X_validation HivData shape: " + str(X_valid.shape))
print("y_validation HivData shape: " + str(y_valid.shape) + "\n")

# print test HivData shape and input shape
print("X_test HivData shape: " + str(X_test.shape))
print("y_test HivData shape: " + str(y_test.shape) + "\n")
print("Model input shape: " + str(input_shape) + "\n")


#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=6082)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


#  _____________________Model setup and 5-fold CV_____________________
# inspiration: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_05_2_kfold.ipynb
if (os.path.exists(path + 'results.csv')):
    print(f"_________Files at {path} was found. If you want to train a new model, delete files in that path_________")
    print()

    eval_df = pd.read_csv(path + "results.csv")
    print(eval_df)

else:

    cv_results = pd.DataFrame(
        columns=['Train Loss', 'Validation Loss', 'Test Loss', 'Train AUC', 'Validation AUC', 'Test AUC'])

    print("______________Training model______________")
    print()


    # Building the model
    model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=2, lr=0.00001)

    print(model.summary())

    # Setup callbacks
    filecp = path + "_bestweights_trial_" + ".hdf5"
    filecsv = path + "_loss_curve_"  + ".csv"
    callbacks = [TerminateOnNaN(),
                 LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush()),
                 EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto'),
                 ModelCheckpoint(filecp, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"),
                 CSVLogger(filecsv)]

    # Train model
    datagen = ImageDataGenerator()
    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  epochs=nb_epoch, steps_per_epoch=X_train.shape[0] / batch_size,
                                  verbose=verbose,
                                  validation_data=(X_valid, y_valid),
                                  callbacks=callbacks)

    # Visualize loss curve
    hist_df = cs_keras_to_seaborn(hist)
    cs_make_plots(hist_df, filename=path)

    # Save model and history
    hist = hist.history
    model.save(path + modelName)
    pickle_out = open(path + modelName + "_History" + ".pickle", "wb")
    pickle.dump(hist, pickle_out)
    pickle_out.close()

    with tf.device('/cpu:0'):
        # Reload best model & compute results
        model.load_weights(filecp)
        cs_compute_results(model, classes=2, df_out=cv_results,
                           train_data=(X_train, y_train),
                           valid_data=(X_valid, y_valid),
                           test_data=(X_test, y_test),
                           filename=path)

    # Calculate results for entire CV
    final_mean = cv_results.mean(axis=0)
    final_std = cv_results.std(axis=0)
    cv_results.to_csv(path + 'results.csv', index=False)

    # Print final results
    print('*** TRIAL RESULTS: ')
    print('*** PARAMETERS TESTED: ' + str(params))
    print(cv_results)
