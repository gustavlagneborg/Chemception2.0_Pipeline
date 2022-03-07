import statistics
import matplotlib.pyplot as plt
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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# varibale setup
random_state = 125
tf.random.set_seed(
    random_state
)

path = "SavedModels/Chemception_test/"
modelName = "Chemception_test"


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
    trainAndValidData = tensorDataPrep(loadPath=DirTrainImg, savePath=DirTrainArray, testOrTrain="Train")

    # Loading individual test data
if (os.path.exists("Data/MolFromSmilesArray/testData.pickle")):
    print("Local test data was found" + "\n")
    pickle_in = open("Data/MolFromSmilesArray/testData.pickle","rb")
    testData = pickle.load(pickle_in)
else:
    testData = tensorDataPrep(loadPath=DirTestImg, savePath=DirTestArray, testOrTrain="Test")

X_train_and_valid = np.array(list(trainAndValidData.iloc[:, 0].values))
y_train_and_valid = trainAndValidData.iloc[:, 1].values

X_test = np.array(list(testData.iloc[:, 0].values))
y_test = testData.iloc[:, 1].values.reshape(-1,1)


#  _____________________Check if GPU is available_____________________
print("Num GPUs Available: ", str(len(tf.config.list_physical_devices('GPU'))) + "\n")
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#  _____________________Model setup and 5-fold CV_____________________
# inspiration: https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_05_2_kfold.ipynb

model_train_accuarcy = []
model_val_accuarcy = []
model_test_accuarcy = []
model_train_AUC = []
model_val_AUC = []
model_test_AUC = []

if (os.path.exists(path + modelName + "_Evaluation_df" + ".pickle")):
    print(f"_________Files at {path} was found. If you want to train a new model, delete files in that path_________")
    print()

    pickle_in = open(path + modelName + "_Evaluation_df" + ".pickle","rb")
    eval_df = pickle.load(pickle_in)
    print(eval_df)

else:
    kfolds = 5
    input_shape = X_train_and_valid.shape[1:]
    kf = StratifiedKFold(kfolds, shuffle=True, random_state=random_state) 
    fold = 0

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
        input_img = Input(shape=input_shape)
        x = Inception0(input_img)
        x = Inception(x)
        x = Inception(x)
        od=int(x.shape[1])
        x = MaxPooling2D(pool_size=(od,od), strides=(1,1))(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        output = Dense(y_train_cv.shape[1], activation='sigmoid')(x)
        model = Model(inputs=input_img, outputs=output, name=name)

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
                                epochs=50,
                                validation_data=(X_valid_cv, y_valid_cv),
                                callbacks=[reduce_lr])

        # train predictions
        train_y_pred_cv = model.predict(X_train_cv)
        train_y_pred_cv = np.round(train_y_pred_cv)

        # validiation predictions
        valid_y_pred_cv = model.predict(X_valid_cv)
        valid_y_pred_cv = np.round(valid_y_pred_cv)

        # test predictions (individual testset)
        test_y_pred = model.predict(X_test)
        test_y_pred = np.round(test_y_pred)

        # Calculating model Accuracy
        model_train_accuarcy.append(history.history["accuracy"][1])
        model_val_accuarcy.append(history.history["val_accuracy"][1])
        model_test_accuarcy.append(accuracy_score(y_test, test_y_pred))

        # Calculating model AUC
        model_train_AUC.append(roc_auc_score(y_train_cv, train_y_pred_cv))
        model_val_AUC.append(roc_auc_score(y_valid_cv, valid_y_pred_cv))
        model_test_AUC.append(roc_auc_score(y_test, test_y_pred))

        # Save model and history
        hist = history.history
        model.save(path + name)
        pickle_out = open(path + name + "_History" + ".pickle","wb")
        pickle.dump(hist, pickle_out)
        pickle_out.close()

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
     
    



