import statistics
from numpy import std
from tensorflow.python.keras.activations import softmax

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
import seaborn as sns
from itertools import combinations



def cs_compute_results(model, classes=None, train_data=None, valid_data=None, test_data=None, df_out=None, filename=None):
    
    # Evaluate results on training set
    X_tmp = train_data[0]
    y_tmp = train_data[1]    
    loss_train = model.evaluate(X_tmp, y_tmp, verbose=0)
    
    if classes == 1:
        rmse_train = np.sqrt(loss_train)
    elif classes == 2:
        y_preds_train = model.predict(X_tmp, batch_size=128)
        auc_train = cs_auc(y_tmp, y_preds_train)
    elif classes > 2:
        y_preds_train = model.predict(X_tmp, batch_size=128)
        auc_train = cs_multiclass_auc(y_tmp, y_preds_train)
    else:
        raise(Exception('Error in determine problem type'))
        
    # Evaluate results on validation set    
    X_tmp = valid_data[0]
    y_tmp = valid_data[1]
    loss_valid = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_valid = np.sqrt(loss_valid)
    elif classes == 2:
        y_preds_valid = model.predict(X_tmp, batch_size=128)
        auc_valid = cs_auc(y_tmp, y_preds_valid)
    elif classes > 2:
        y_preds_valid = model.predict(X_tmp, batch_size=128)
        auc_valid = cs_multiclass_auc(y_tmp, y_preds_valid)
    else:
        raise(Exception('Error in determine problem type'))
    
    # Evaluate results on test set
    X_tmp = test_data[0]
    y_tmp = test_data[1]    
    loss_test = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_test = np.sqrt(loss_test)
    elif classes == 2:
        y_preds_test = model.predict(X_tmp, batch_size=128)
        auc_test = cs_auc(y_tmp, y_preds_test)

        data = {'y_Actual': np.argmax(y_tmp, axis=1),
                'y_Predicted': np.argmax(y_preds_test, axis=1)
                }

        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

        matrix = sns.heatmap(confusion_matrix, annot=True)
        matrix.figure.savefig(os.path.join(filename, 'ConfusionMatrix.png'), dpi=400)
        plt.show()

    elif classes > 2:
        y_preds_test = model.predict(X_tmp, batch_size=128)
        auc_test = cs_multiclass_auc(y_tmp, y_preds_test)
    else:
        raise(Exception('Error in determine problem type'))
    
    if classes == 1:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid))
        print("FINAL TST_LOSS: %.3f"%(loss_test))
        print("FINAL TRA_RMSE: %.3f"%(rmse_train))
        print("FINAL VAL_RMSE: %.3f"%(rmse_valid))
        print("FINAL TST_RMSE: %.3f"%(rmse_test))
        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, rmse_train, rmse_valid, rmse_test]
    else:
        print()
        print("FINAL TRA_LOSS: ")
        print(loss_train)

        print("FINAL TST_LOSS ")
        print(loss_valid)

        print("FINAL TRA_LOSS: ")
        print(loss_test)

        print("FINAL TRA_AUC: %.3f"%(auc_train))
        print("FINAL VAL_AUC: %.3f"%(auc_valid))
        print("FINAL TST_AUC: %.3f"%(auc_test))
        print()

        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, auc_train, auc_valid, auc_test]




def cs_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc

def cs_keras_to_seaborn(history):
    tmp_frame = pd.DataFrame(history.history)
    keys = list(history.history.keys())
    features = [x for x in keys if "val_" not in x and "val_" + x in keys]
    cols = ['epoch', 'phase'] + features
    output_df = pd.DataFrame(columns=cols)
    epoch = 1
    for i in range(len(tmp_frame)):
        new_row = [epoch, 'train'] + [tmp_frame.loc[i, f] for f in features]
        output_df.loc[len(output_df)] = new_row
        new_row = [epoch, 'validation'] + [tmp_frame.loc[i, "val_" + f] for f in features]
        output_df.loc[len(output_df)] = new_row
        epoch += 1
    return output_df

def cs_make_plots(hist_df, filename=None):
    fig, axes = plt.subplots(1, 1)
    sns.pointplot(x='epoch', y='loss', hue='phase', data=hist_df, ax=axes)
    axes.set_title('Loss Curve', fontdict={'size': 20})
    axes.set_ylim(np.min(hist_df['loss']), np.max(hist_df['loss']))
    plt.show()
    fig.savefig(os.path.join(filename, 'LossCurve.png'))

def cs_multiclass_auc(y_true, y_pred):
    n = y_pred.shape[1]
    auc_dict = dict()
    for pair in combinations(range(n), 2):
        subset = [i for i in range(len(y_true)) if 1 in [y_true[i, pair[0]], y_true[i, pair[1]]]]
        y_true_temp = y_true[subset]
        y_pred_temp = y_pred[subset]
        y_pred_temp = y_pred_temp[:, [pair[0], pair[1]]]
        y_pred_temp = softmax(y_pred_temp)
        auc_dict[pair] = roc_auc_score(y_true_temp[:, pair[1]], y_pred_temp[:, 1])
    total = 0.0
    for key in auc_dict.keys():
        total += auc_dict[key]
    total /= len(list(combinations(range(n), 2)))
    return total



