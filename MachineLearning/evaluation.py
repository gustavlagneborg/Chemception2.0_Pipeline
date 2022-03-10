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

def cs_compute_results(model, classes=None, train_data=None, valid_data=None, test_data=None, df_out=None):
    
    # Evaluate results on training set
    X_tmp = train_data[0]
    y_tmp = train_data[1]    
    loss_train = model.evaluate(X_tmp, y_tmp, verbose=0)
    print(loss_train)
    
    if classes == 1:
        rmse_train = np.sqrt(loss_train[1])
    elif classes == 2:
        y_preds_train = model.predict(X_tmp)
        auc_train = cs_auc(y_tmp, y_preds_train)
    elif classes > 2:
        y_preds_train = model.predict(X_tmp)
        auc_train = cs_multiclass_auc(y_tmp, y_preds_train)
    else:
        raise(Exception('Error in determine problem type'))
        
    # Evaluate results on validation set    
    X_tmp = valid_data[0]
    y_tmp = valid_data[1]
    loss_valid = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_valid = np.sqrt(loss_valid[1])
    elif classes == 2:
        y_preds_valid = model.predict(X_tmp)
        auc_valid = cs_auc(y_tmp, y_preds_valid)
    elif classes > 2:
        y_preds_valid = model.predict(X_tmp)
        auc_valid = cs_multiclass_auc(y_tmp, y_preds_valid)
    else:
        raise(Exception('Error in determine problem type'))
    
    # Evaluate results on test set
    X_tmp = test_data[0]
    y_tmp = test_data[1]    
    loss_test = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_test = np.sqrt(loss_test[1])
    elif classes == 2:
        y_preds_test = model.predict(X_tmp)
        auc_test = cs_auc(y_tmp, y_preds_test)
    elif classes > 2:
        y_preds_test = model.predict(X_tmp)
        auc_test = cs_multiclass_auc(y_tmp, y_preds_test)
    else:
        raise(Exception('Error in determine problem type'))
    
    if classes == 1:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train[1]))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid[1]))
        print("FINAL TST_LOSS: %.3f"%(loss_test[1]))
        print("FINAL TRA_RMSE: %.3f"%(rmse_train))
        print("FINAL VAL_RMSE: %.3f"%(rmse_valid))
        print("FINAL TST_RMSE: %.3f"%(rmse_test))
        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, rmse_train, rmse_valid, rmse_test]
    else:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid))
        print("FINAL TST_LOSS: %.3f"%(loss_test))
        print("FINAL TRA_AUC: %.3f"%(auc_train))
        print("FINAL VAL_AUC: %.3f"%(auc_valid))
        print("FINAL TST_AUC: %.3f"%(auc_test))
        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, auc_train, auc_valid, auc_test]


def cs_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc
