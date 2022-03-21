from Preprocessing.DataPrep import *
from Preprocessing.ImageCreation import *
import os.path
from tdc.single_pred import ADME
from sklearn.model_selection import train_test_split
from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *
import statistics

import pandas as pd
from numpy import std
from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
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

random_state = 125
print(tf.__version__)

# configuration
tf.random.set_seed(
    random_state
)

loadPath = "../SavedModels/BestModel/"
path = "SmilesColorLipo/"
modelName = "TransferLearningLipo"
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

# preprocessing
train_file_exists = os.path.exists('../LipophilicityData/df_train_preprocessed.csv')
test_file_exists = os.path.exists('../LipophilicityData/df_test_preprocessed.csv')

if (train_file_exists == False and test_file_exists == False):
    # get dataset
    data = ADME(name='Lipophilicity_AstraZeneca')
    df = data.get_data()
    # df.drop(['Drug'], axis=1, inplace=True)
    print(df.head())
    df.rename(columns={'Drug_ID': 'MolName', 'Drug': 'SMILES', 'Y': 'Lipophilicity'}, inplace=True)

    # clean up
    # add column flagging compounds with uncommon atom types
    numCpdsBefore = df.shape[0]

    # remove compounds with uncommon atoms types
    df['OnlyCommonAtms'] = df.SMILES.progress_apply(commonAtoms)
    df = df.loc[(df['OnlyCommonAtms'] == True)]

    # removing salts, cleaning up SMILES
    df['SMILES'] = df.SMILES.progress_apply(stripSalts)

    # neutralize
    df['SMILES'] = df.SMILES.progress_apply(neutralize_atoms)  # clean SMILES

    # calculate and filter on three simple descriptors (RDkit)
    df['desc'] = df.SMILES.progress_apply(calc_3_descriptors)
    desc_cols = ['MW', 'Rotors', 'Heavys']
    df[desc_cols] = df.desc.to_list()
    df.drop("desc", axis=1, inplace=True)

    # filter on simple properties
    df = df.loc[(df['Heavys'] > 5) &
                (df['Heavys'] < 50) &
                (df['Rotors'] < 18) &
                (df['MW'] > 100) &
                (df['MW'] < 900)  # &
        # (df['OnlyCommonAtms'] == True)
                ]

    # drop columns from dataframe
    df.drop(["OnlyCommonAtms", 'Heavys', 'MW', 'Rotors'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    numCpdsAfter = df.shape[0]
    print(
        f"Starting with {numCpdsBefore} compounds in entire dataset. This many remaining {numCpdsAfter} after filtering.")

    # split into 1/6th testset and 5/6 for train
    training_data, testing_data = train_test_split(df, test_size=1 / 6, random_state=random_state)
    training_data = training_data.reset_index(drop=True)
    testing_data = testing_data.reset_index(drop=True)

    # Save preprocessed test and train dataset
    training_data.to_csv(path_or_buf="../LipophilicityData/df_train_preprocessed.csv")
    testing_data.to_csv(path_or_buf="../LipophilicityData/df_test_preprocessed.csv")
else:
    print("Found local preprocessed dataframe")
    training_data = pd.read_csv("../LipophilicityData/df_train_preprocessed.csv", index_col=[0]).reset_index(drop=True)
    testing_data = pd.read_csv("../LipophilicityData/df_test_preprocessed.csv", index_col=[0]).reset_index(drop=True)

# image creation
imageType = "SmilesColor"

if imageType == "SmilesColor":
    for fname in os.listdir("../LipophilicityImages/SmilesColorImages/Train"):
        if fname.endswith('.png'):
            print("png files already exists in specified path - Smiles Color LipophilicityImages")
            break
    else:
        # train images
        generateImageSMILEColor(path="../LipophilicityImages/SmilesColorImages/Train/", compoundList=training_data, HIV_activity="regression", withChars=False)

        # test images
        generateImageSMILEColor(path="../LipophilicityImages/SmilesColorImages/Test/", compoundList=testing_data, HIV_activity="regression", withChars=False)

# Transfer learning
DirTrainImg = "LipophilicityImages/SmilesColorImages/Train/"
DirTestImg = "LipophilicityImages/SmilesColorImages/Test/"
DirTensorArray = "LipophilicityData/SmilesColorArray/"

# Loading trainig LipophilicityData
if os.path.exists("../LipophilicityData/SmilesColorArray/X_Train.pickle"):
    print("Local train LipophilicityData was found" + "\n")
    pickle_in = open("../LipophilicityData/SmilesColorArray/X_Train.pickle", "rb")
    X_train_and_valid = pickle.load(pickle_in)

    pickle_in = open("../LipophilicityData/SmilesColorArray/y_Train.pickle", "rb")
    y_train_and_valid = pickle.load(pickle_in)

else:
    print("Producing train LipophilicityData!" + "\n")
    X_train_and_valid, y_train_and_valid = tensorDataPreRegression(loadPath=DirTrainImg, savePath=DirTensorArray, testOrTrain="Train")
    print("Done!")

    # Loading individual test LipophilicityData
if os.path.exists("../LipophilicityData/SmilesColorArray/X_Test.pickle"):
    print("Local test LipophilicityData was found" + "\n")
    pickle_in = open("../LipophilicityData/SmilesColorArray/X_Test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("../LipophilicityData/SmilesColorArray/y_Test.pickle", "rb")
    y_test = pickle.load(pickle_in)

else:
    print("Producing test LipophilicityData!" + "\n")
    X_test, y_test = tensorDataPreRegression(loadPath=DirTestImg, savePath=DirTensorArray, testOrTrain="Test")
    print("Done!")

# test and valid split
X_train, X_valid, y_train, y_valid = train_test_split(
                                        X_train_and_valid,
                                        y_train_and_valid,
                                        test_size=0.2,
                                        random_state=random_state,
                                        shuffle=True)

# print LipoData shapes before oversampling
print("LipoData shapes: ")
print("X_train HivData shape: " + str(X_train.shape))
print("y_train HivData shape: " + str(y_train.shape) + "\n")

print("X_validation HivData shape: " + str(X_valid.shape))
print("y_validation HivData shape: " + str(y_valid.shape) + "\n")

y_train = y_train.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
input_shape = X_train.shape[1:]

# show sample of a molecule
v = X_train_and_valid[0]
plt.imshow(v[:,:,:3])
plt.show()


#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6082)])
    except RuntimeError as e:
        print(e)

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# create model with preloaded weights
results = pd.DataFrame(columns=['Train Loss', 'Validation Loss', 'Test Loss', 'Train RMSE', 'Validation RMSE', 'Test RMSE'])

# load pretrained model
filecp = loadPath + "_bestweights_trial_" + ".hdf5"
model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=1, lr=0.001)
submodel.load_weights(filecp, by_name=True)

for layer in submodel.layers:
    layer.trainable = False

tf_model = Sequential()
tf_model.add(submodel)
tf_model.add(Dense(units=1, activation='linear'))

optimizer = RMSprop(lr=0.001)
tf_model.compile(optimizer=optimizer, loss="mean_squared_error")

print(tf_model.summary())

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
hist = tf_model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                              epochs=nb_epoch, steps_per_epoch=X_train.shape[0] / batch_size,
                              verbose=verbose,
                              validation_data=(X_valid, y_valid),
                              callbacks=callbacks)

# Visualize loss curve
hist_df = cs_keras_to_seaborn(hist)
cs_make_plots(hist_df, filename=path)

# Save model and history
hist = hist.history
tf_model.save(path + modelName)
pickle_out = open(path + modelName + "_History" + ".pickle", "wb")
pickle.dump(hist, pickle_out)
pickle_out.close()

# Reload best model & compute results
tf_model.load_weights(filecp)
cs_compute_results(tf_model, classes=1, df_out=results,
                   train_data=(X_train, y_train),
                   valid_data=(X_valid, y_valid),
                   test_data=(X_test, y_test),
                   filename=path)

# Calculate results for entire CV
final_mean = results.mean(axis=0)
final_std = results.std(axis=0)
results.to_csv(path + 'results.csv', index=False)

# Print final results
print('*** TRIAL RESULTS: ')
print('*** PARAMETERS TESTED: ' + str(params))
print(results)