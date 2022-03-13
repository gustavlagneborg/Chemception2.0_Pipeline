from Preprocessing.DataPrep import *
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)

path = "SavedModels/T1_ChemOriginal/"
dataPath = "Data/ChemOriginalArray/"
modelName = "T1_MolSmiles"
batch_size = 32
nb_epoch = 2
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
if os.path.exists(dataPath + 'df_train_preprocessed.pkl'):
    print("Local data was found!")
    training_data = pd.read_pickle(dataPath + "df_train_preprocessed.pkl")
    testing_data = pd.read_pickle(dataPath + "df_test_preprocessed.pkl")

else:
    print("Producing data for ChemOriginal!")
    training_data = pd.read_csv("data/df_train_preprocessed.csv", index_col=[0]).reset_index(drop=True)
    testing_data = pd.read_csv("data/df_test_preprocessed.csv", index_col=[0]).reset_index(drop=True)

    training_data["mol"] = training_data["SMILES"].apply(Chem.MolFromSmiles)
    testing_data["mol"] = testing_data["SMILES"].apply(Chem.MolFromSmiles)


    def vectorize(mol):
        return chemcepterize_mol(mol, embed=20.0, res=0.5)


    training_data["molimage"] = training_data["mol"].apply(vectorize)
    testing_data["molimage"] = testing_data["mol"].apply(vectorize)
    print("Done!")

    training_data.to_pickle(dataPath + "df_train_preprocessed.pkl")
    testing_data.to_pickle(dataPath + "df_test_preprocessed.pkl")

X_train_and_valid = np.array(list(training_data["molimage"].values))
y_train_and_valid = training_data["HIV_active"].values

X_test = np.array(list(testing_data["molimage"].values))
y_test = testing_data["HIV_active"].values.reshape(-1, 1)

input_shape = X_train_and_valid.shape[1:]

#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
  except RuntimeError as e:
    print(e)

if os.path.exists(path + 'results.csv'):
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

        X_train_cv = np.asarray(X_train_and_valid[train])
        y_train_cv = np.asarray(y_train_and_valid[train]).reshape(-1, 1)
        X_valid_cv = np.asarray(X_train_and_valid[test])
        y_valid_cv = np.asarray(y_train_and_valid[test]).reshape(-1, 1)

        # Building the model
        model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=1)

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
            cs_compute_results(model, classes=1, df_out=cv_results,
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

