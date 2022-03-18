import statistics
from numpy import std
from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

import keras
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
import sys

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

print(tf.__version__)

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)

path = "../SavedModels/MolSmiles/"
modelName = "MolSmiles"
batch_size = 64
nb_epoch = 100
verbose = 1

# change depending on image, 180 for mol images, 0 for others
rotation_range = 180

params = {
    'conv1_units': 32,
    'conv2_units': 32,
    'conv3_units': 32,
    'conv4_units': 32,
    'conv5_units': 32,
    'conv6_units': 32,
    'num_block1': 1,
    'num_block2': 1,
    'num_block3': 1,
    'dropval': 0,
}

# _____________________load or create HivData with if statement _____________________
DirTrainImg = "HIVImages/MolFromSmilesImages/Train/"
DirTestImg = "HIVImages/MolFromSmilesImages/Test/"
DirTensorArray = "Data/MolFromSmilesArray/"

# Loading trainig HivData
if os.path.exists("../LipophilicityData/MolFromSmilesArray/X_Train.pickle"):
    print("Local train HivData was found" + "\n")
    pickle_in = open("../HivData/MolFromSmilesArray/X_Train.pickle", "rb")
    X_train_and_valid = pickle.load(pickle_in)

    pickle_in = open("../HivData/MolFromSmilesArray/y_Train.pickle", "rb")
    y_train_and_valid = pickle.load(pickle_in)

else:
    print("Producing train HivData!" + "\n")
    X_train_and_valid, y_train_and_valid = tensorDataPrep(loadPath=DirTrainImg, savePath=DirTensorArray,
                                                          testOrTrain="Train")
    print("Done!")

    # Loading individual test HivData
if os.path.exists("../LipophilicityData/MolFromSmilesArray/X_Test.pickle"):
    print("Local test HivData was found" + "\n")
    pickle_in = open("../HivData/MolFromSmilesArray/X_Test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("../HivData/MolFromSmilesArray/y_Test.pickle", "rb")
    y_test = pickle.load(pickle_in)


else:
    print("Producing test HivData!" + "\n")
    X_test, y_test = tensorDataPrep(loadPath=DirTestImg, savePath=DirTensorArray, testOrTrain="Test")
    print("Done!")

input_shape = X_train_and_valid.shape[1:]

#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5837)])
    except RuntimeError as e:
        print(e)

#  _____________________Model setup and 5-fold CV_____________________
# inspiration: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_05_2_kfold.ipynb
if (os.path.exists(path + 'results.csv')):
    print(f"_________Files at {path} was found. If you want to train a new model, delete files in that path_________")
    print()

    eval_df = pd.read_csv(path + "results.csv")
    print(eval_df)

else:
    kfolds = 5
    kf = StratifiedKFold(kfolds, shuffle=True, random_state=random_state)
    fold = 0
    cv_results = pd.DataFrame(
        columns=['Train Loss', 'Validation Loss', 'Test Loss', 'Train AUC', 'Validation AUC', 'Test AUC'])

    print("______________Training model - 5 fold CV______________")
    print()
    for train, test in kf.split(X_train_and_valid, y_train_and_valid):
        fold += 1
        print(f"Fold #{fold}")

        name = modelName + "_" + str(fold)

        # train HivData
        X_train_cv = np.asarray(X_train_and_valid[train])
        y_train_cv = np.asarray(y_train_and_valid[train])
        y_train_cv = tf.one_hot(y_train_cv.flatten(), depth=2)

        # validation HivData
        X_valid_cv = np.asarray(X_train_and_valid[test])
        y_valid_cv = np.asarray(y_train_and_valid[test])
        y_valid_cv = tf.one_hot(y_valid_cv.flatten(), depth=2)

        # Building the model
        model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=2)

        if fold == 1:
            print(model.summary())

        # Setup callbacks
        filecp = path + "_bestweights_trial_" + str(fold) + ".hdf5"
        filecsv = path + "_loss_curve_" + str(fold) + ".csv"
        callbacks = [TerminateOnNaN(),
                     LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush()),
                     EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto'),
                     ModelCheckpoint(filecp, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"),
                     CSVLogger(filecsv)]

        # Train model
        datagen = ImageDataGenerator(rotation_range=rotation_range, fill_mode='constant', cval=0.)
        history = model.fit_generator(datagen.flow(X_train_cv, y_train_cv, batch_size=batch_size),
                                      epochs=nb_epoch, steps_per_epoch=X_train_cv.shape[0] / batch_size,
                                      verbose=verbose,
                                      validation_data=(X_valid_cv, y_valid_cv),
                                      callbacks=callbacks)

        # Save model and history
        hist = history.history
        model.save(path + name)
        pickle_out = open(path + name + "_History" + ".pickle", "wb")
        pickle.dump(hist, pickle_out)
        pickle_out.close()

        with tf.device('/cpu:0'):
            # Reload best model & compute results
            model.load_weights(filecp)
            cs_compute_results(model, classes=2, df_out=cv_results,
                               train_data=(X_train_cv, y_train_cv),
                               valid_data=(X_valid_cv, y_valid_cv),
                               test_data=(X_test, y_test))

    # Calculate results for entire CV
    final_mean = cv_results.mean(axis=0)
    final_std = cv_results.std(axis=0)
    cv_results.to_csv(path + 'results.csv', index=False)

    # Print final results
    print('*** TRIAL RESULTS: ' + str(fold))
    print('*** PARAMETERS TESTED: ' + str(params))
    print(cv_results)
