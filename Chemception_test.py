import statistics
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
# inspiration: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_05_2_kfold.ipynb

input_shape = X_train.shape[1:]
kf = StratifiedKFold(5, shuffle=True, random_state=42) 
fold = 0
model_accuarcy = []
model_AUC = []

for train, test in kf.split(X_train, y_train):
    fold+=1
    print(f"Fold #{fold}")
    
    modelName = "Chemception_test_" + str(fold)

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
    output = Dense(y_train_cv.shape[1], activation='sigmoid')(x)
    model = Model(inputs=input_img, outputs=output, name=modelName)

    optimizer = Adam(lr=0.00025)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    tf.config.run_functions_eagerly(True)

    generator = ImageDataGenerator(rotation_range=180,
                               width_shift_range=0.1, height_shift_range=0.1,
                               fill_mode="constant", cval = 0,
                               horizontal_flip=True, vertical_flip=True, data_format='channels_last')

    batch_size=32
    g = generator.flow(X_train_cv, y_train_cv, batch_size=batch_size, shuffle=True)
    steps_per_epoch = 10000/batch_size

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=1e-6, verbose=1)

    history = model.fit_generator(g,
                              steps_per_epoch=len(X_train_cv)//batch_size,
                              epochs=2,
                              validation_data=(X_test_cv, y_test_cv),
                              callbacks=[reduce_lr])

    # Accuracy and AUC
    y_pred_cv = model.predict(X_test_cv)
    y_pred_cv = np.round(y_pred_cv)
    model_accuarcy.append(accuracy_score(y_test_cv, y_pred_cv))
    model_AUC.append(roc_auc_score(y_test_cv, y_pred_cv))

    # Save model and history
    path = "SavedModels/Chemception_test/"
    hist = history.history

    model.save(path + modelName)
    pickle_out = open(path + modelName + "_History" + ".pickle","wb")
    pickle.dump(hist, pickle_out)
    pickle_out.close()

print("__________Validation accuracy and AUC__________")
print("Accuracy: ")
print(model_accuarcy)
print(statistics.mean(model_accuarcy))
print()

print("AUC: ")
print(model_AUC)
print(statistics.mean(model_AUC))
print()

#_____________________evaluation_____________________
# train/validation data

# test data
