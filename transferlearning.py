from Preprocessing.DataPrep import *
from Preprocessing.ImageCreation import *
import os.path
from tdc.single_pred import ADME
from sklearn.model_selection import train_test_split
from MachineLearning.CNNDesignAndSetup import *
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *

random_state = 125
print(tf.__version__)

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)

path = "SavedModels/SmilesColor/"
modelName = "Transfer learning"
batch_size = 32
nb_epoch = 100
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
train_file_exists = os.path.exists('LipophilicityData/df_train_preprocessed.csv')
test_file_exists = os.path.exists('LipophilicityData/df_test_preprocessed.csv')

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
    training_data.to_csv(path_or_buf="LipophilicityData/df_train_preprocessed.csv")
    testing_data.to_csv(path_or_buf="LipophilicityData/df_test_preprocessed.csv")
else:
    print("Found local preprocessed dataframe")
    training_data = pd.read_csv("LipophilicityData/df_train_preprocessed.csv", index_col=[0]).reset_index(drop=True)
    testing_data = pd.read_csv("LipophilicityData/df_test_preprocessed.csv", index_col=[0]).reset_index(drop=True)

# image creation
imageType = "SmilesColor"

if imageType == "SmilesColor":
    for fname in os.listdir("LipophilicityImages/SmilesColorImages/Train"):
        if fname.endswith('.png'):
            print("png files already exists in specified path - Smiles Color LipophilicityImages")
            break
    else:
        # train images
        generateImageSMILEColor(path="LipophilicityImages/SmilesColorImages/Train/", compoundList=training_data, HIV_activity="regression", withChars=False)

        # test images
        generateImageSMILEColor(path="LipophilicityImages/SmilesColorImages/Test/", compoundList=testing_data, HIV_activity="regression", withChars=False)

# Transfer learning
DirTrainImg = "LipophilicityImages/SmilesColorImages/Train/"
DirTestImg = "LipophilicityImages/SmilesColorImages/Test/"
DirTensorArray = "LipophilicityData/SmilesColorArray/"

# Loading trainig LipophilicityData
if os.path.exists("LipophilicityData/SmilesColorArray/X_Train.pickle"):
    print("Local train LipophilicityData was found" + "\n")
    pickle_in = open("LipophilicityData/SmilesColorArray/X_train.pickle", "rb")
    X_train_and_valid = pickle.load(pickle_in)

    pickle_in = open("LipophilicityData/SmilesColorArray/y_Train.pickle", "rb")
    y_train_and_valid = pickle.load(pickle_in)

else:
    print("Producing train LipophilicityData!" + "\n")
    X_train_and_valid, y_train_and_valid = tensorDataPreRegression(loadPath=DirTrainImg, savePath=DirTensorArray, testOrTrain="Train")
    print("Done!")

    # Loading individual test LipophilicityData
if os.path.exists("LipophilicityData/SmilesColorArray/X_Test.pickle"):
    print("Local test LipophilicityData was found" + "\n")
    pickle_in = open("LipophilicityData/SmilesColorArray/X_Test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("LipophilicityData/SmilesColorArray/y_Test.pickle", "rb")
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
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10444)])
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

filecp = "path for pre trained weights"
model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=1)
print(model.summary())
model.load_weights(filecp)

# train..

cs_compute_results(model, classes=2, df_out=results,
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