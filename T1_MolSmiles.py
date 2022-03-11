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
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

print(tf.__version__)

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)


path = "SavedModels/T1_MolSmiles/"
modelName = "T1_MolSmiles"
batch_size = 32
nb_epoch = 5 #100
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

# _____________________load or create data with if statement _____________________
DirTrainImg = "Images/MolFromSmilesImages/Train/"
DirTestImg = "Images/MolFromSmilesImages/Test/"
DirTrainArray = "Data/MolFromSmilesArray/"
DirTestArray = "Data/MolFromSmilesArray/"

# Loading trainig data
if (os.path.exists("Data/MolFromSmilesArray/trainData.pickle")):
    print("Local train data was found" + "\n")
    pickle_in = open("Data/MolFromSmilesArray/trainData.pickle","rb")
    trainAndValidData = pickle.load(pickle_in)

else:
    print("Producing train data!" + "\n")
    trainAndValidData = tensorDataPrep(loadPath=DirTrainImg, savePath=DirTrainArray, testOrTrain="Train")
    print("Done!")

    # Loading individual test data
if (os.path.exists("Data/MolFromSmilesArray/testData.pickle")):
    print("Local test data was found" + "\n")
    pickle_in = open("Data/MolFromSmilesArray/testData.pickle","rb")
    testData = pickle.load(pickle_in)
else:
    print("Producing test data!" + "\n")
    testData = tensorDataPrep(loadPath=DirTestImg, savePath=DirTestArray, testOrTrain="Test")
    print("Done!")

X_train_and_valid = np.array(list(trainAndValidData.iloc[:, 0].values))
y_train_and_valid = trainAndValidData.iloc[:, 1].values


X_test = np.array(list(testData.iloc[:, 0].values))
y_test = testData.iloc[:, 1].values.reshape(-1,1)

#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
  except RuntimeError as e:
    print(e)

#  _____________________Model setup and 5-fold CV_____________________
# inspiration: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_05_2_kfold.ipynb

model_train_accuarcy = []
model_val_accuarcy = []
model_test_accuarcy = []
model_train_AUC = []
model_val_AUC = []
model_test_AUC = []
input_shape = X_train_and_valid.shape[1:]

if (os.path.exists(path + 'results.csv')):
    print(f"_________Files at {path} was found. If you want to train a new model, delete files in that path_________")
    print()

    eval_df = pd.read_csv(path + "results.csv")

    """pickle_in = open(path + modelName + "_Evaluation_df" + ".pickle","rb")
    eval_df = pickle.load(pickle_in)"""
    print(eval_df)

else:
    kfolds = 5
    kf = StratifiedKFold(kfolds, shuffle=True, random_state=random_state) 
    fold = 0
    cv_results = pd.DataFrame(columns=['Train Loss', 'Validation Loss', 'Test Loss', 'Train AUC', 'Validation AUC', 'Test AUC'])

    print("______________Training model - 5 fold CV______________")
    print()
    for train, test in kf.split(X_train_and_valid, y_train_and_valid):
        fold+=1
        print(f"Fold #{fold}")

        name = modelName + "_" + str(fold)

        X_train_cv = np.asarray(X_train_and_valid[train])
        y_train_cv = np.asarray(y_train_and_valid[train]).reshape(-1,1)
        X_valid_cv = np.asarray(X_train_and_valid[test])
        y_valid_cv = np.asarray(y_train_and_valid[test]).reshape(-1,1)
        
        # Building the model
        model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=1)

        if fold == 1:
            print(model.summary())

        # Setup callbacks
        filecp = path+"_bestweights_trial_"+str(fold)+".hdf5"
        filecsv = path+"_loss_curve_"+str(fold)+".csv"
        callbacks = [TerminateOnNaN(),
                     LambdaCallback(on_epoch_end=lambda epoch,logs: sys.stdout.flush()),
                     EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='auto'),
                     ModelCheckpoint(filecp, monitor="val_loss", verbose=1, save_best_only=True, mode="auto"),
                     CSVLogger(filecsv)]
        
        # Train model
        datagen = ImageDataGenerator(rotation_range=rotation_range, fill_mode='constant', cval=0.)
        history = model.fit_generator(datagen.flow(X_train_cv, y_train_cv, batch_size=batch_size),
                                    epochs=nb_epoch, steps_per_epoch=X_train_cv.shape[0]/batch_size,
                                    verbose=verbose,
                                    validation_data=(X_valid_cv, y_valid_cv),
                                    callbacks=callbacks)

        # Save model and history
        hist = history.history
        model.save(path + name)
        pickle_out = open(path + name + "_History" + ".pickle","wb")
        pickle.dump(hist, pickle_out)
        pickle_out.close()

        with tf.device('/cpu:0'):
            # Reload best model & compute results
            model.load_weights(filecp)
            cs_compute_results(model, classes=1, df_out=cv_results,
                            train_data=(X_train_cv, y_train_cv),
                            valid_data=(X_valid_cv, y_valid_cv),
                            test_data=(X_test,y_test))

    # Calculate results for entire CV
    final_mean = cv_results.mean(axis=0)
    final_std = cv_results.std(axis=0)
    cv_results.to_csv(path + 'results.csv', index=False)
    
    # Print final results
    print('*** TRIAL RESULTS: '+str(fold))
    print('*** PARAMETERS TESTED: '+str(params))
    print(cv_results)

    """print(('train_loss: %.3f +/- %.3f, train_auc: %.3f +/- %.3f, val_loss: %.3f +/- %.3f, val_auc: %.3f +/- %.3f, test_loss: %.3f +/- %.3f, test_auc: %.3f +/- %.3f')
              %(final_mean[0], final_std[0], final_mean[3], final_std[3],
                final_mean[1], final_std[1], final_mean[4], final_std[4],
                final_mean[2], final_std[2], final_mean[5], final_std[5]))"""

    """ model_loaded = keras.models.load_model(path + name)
        # train predictions
        train_y_pred_cv = model_loaded.predict(X_train_cv)
        train_y_pred_cv = np.round(train_y_pred_cv)

        # validiation predictions
        valid_y_pred_cv = model_loaded.predict(X_valid_cv)
        valid_y_pred_cv = np.round(valid_y_pred_cv)

        # test predictions (individual testset)
        test_y_pred = model_loaded.predict(X_test)
        test_y_pred = np.round(test_y_pred)

        # Calculating model Accuracy
        model_train_accuarcy.append(history.history["accuracy"][1])
        model_val_accuarcy.append(history.history["val_accuracy"][1])
        model_test_accuarcy.append(accuracy_score(y_test, test_y_pred))

        # Calculating model AUC
        model_train_AUC.append(roc_auc_score(y_train_cv, train_y_pred_cv))
        model_val_AUC.append(roc_auc_score(y_valid_cv, valid_y_pred_cv))
        model_test_AUC.append(roc_auc_score(y_test, test_y_pred))

        

    #_____________________evaluation_____________________
    # train/validation/test data

    # Build and save evaluation dataframe
    model_train_accuarcy.append(statistics.mean(model_train_accuarcy))
    model_train_accuarcy.append(statistics.stdev(model_train_accuarcy))
    model_val_accuarcy.append(statistics.mean(model_val_accuarcy))
    model_val_accuarcy.append(statistics.stdev(model_val_accuarcy))
    model_test_accuarcy.append(statistics.mean(model_test_accuarcy))
    model_test_accuarcy.append(statistics.stdev(model_test_accuarcy))
    model_train_AUC.append(statistics.mean(model_train_AUC))
    model_train_AUC.append(statistics.stdev(model_train_AUC))
    model_val_AUC.append(statistics.mean(model_val_AUC))
    model_val_AUC.append(statistics.stdev(model_val_AUC))
    model_test_AUC.append(statistics.mean(model_test_AUC))
    model_test_AUC.append(statistics.stdev(model_test_AUC))

    index = ["CV " + str(i) for i in range(1, kfolds+1)]
    index.append("Average")
    index.append("Standard deviation")

    eval_df = pd.DataFrame()
    eval_df ["index"] = index
    eval_df["Train Accuracy"] = model_train_accuarcy
    eval_df["Validation Accuracy"] = model_val_accuarcy
    eval_df["Test Accuracy"] = model_test_accuarcy
    eval_df["Train AUC"] = model_train_AUC
    eval_df["Validation AUC"] = model_val_AUC
    eval_df["Test AUC"] = model_test_AUC

    eval_df = eval_df.set_index("index")
    print()
    print(f"_______________Evaluation of {modelName}_______________")
    print(eval_df)

    pickle_out = open(path + modelName + "_Evaluation_df" + ".pickle","wb")
    pickle.dump(eval_df, pickle_out)
    pickle_out.close()
    """
    



