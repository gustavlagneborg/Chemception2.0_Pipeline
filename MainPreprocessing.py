from Preprocessing.DataPrep import *
from Preprocessing.ImageCreation import *
import os.path
from sklearn.model_selection import train_test_split

random_state = 125

train_file_exists = os.path.exists('data/df_train_preprocessed.csv')
test_file_exists = os.path.exists('data/df_test_preprocessed.csv')

if (train_file_exists == False and test_file_exists == False):
    # get dataset
    df = load_dataset()

    # clean up
    # add column flagging compounds with uncommon atom types
    numCpdsBefore = df.shape[0]

    # remove compounds with uncommon atoms types
    df['OnlyCommonAtms'] = df.SMILES.progress_apply(commonAtoms)
    df = df.loc[(df['OnlyCommonAtms'] == True)]

    # removing salts, cleaning up SMILES
    df['SMILES'] = df.SMILES.progress_apply(stripSalts) 

    # neutralize
    df['SMILES'] = df.SMILES.progress_apply(neutralize_atoms) # clean SMILES

    # calculate and filter on three simple descriptors (RDkit)
    df['desc'] = df.SMILES.progress_apply(calc_3_descriptors)
    desc_cols = ['MW','Rotors','Heavys']
    df[desc_cols] = df.desc.to_list()
    df.drop("desc",axis=1,inplace=True)

    # filter on simple properties
    df = df.loc[(df['Heavys'] > 5) & 
                (df['Heavys'] < 50) &
                (df['Rotors'] < 18) &
                (df['MW'] > 100) &
                (df['MW'] < 900) #&
                #(df['OnlyCommonAtms'] == True)
                ]

    # drop columns from dataframe
    df.drop(["OnlyCommonAtms",'Heavys','MW','Rotors'], axis=1,inplace=True)
    df = df.reset_index(drop=True)
    numCpdsAfter = df.shape[0]
    print(f"Starting with {numCpdsBefore} compounds in entire dataset. This many remaining {numCpdsAfter} after filtering.")

    # split into 1/6th testset and 5/6 for train
    training_data, testing_data = train_test_split(df, test_size=1/6, random_state=random_state, stratify=df["HIV_active"])
    training_data = training_data.reset_index(drop=True)
    testing_data = testing_data.reset_index(drop=True)

    # oversample trainset
    active=training_data["HIV_active"].value_counts()[1]
    inactive=training_data["HIV_active"].value_counts()[0]
    print(f"Class balance in training data before oversampling - HIV_active: {active} HIV_inactive {inactive}.")

    training_data = oversample(training_data, "HIV_active")

    active=training_data["HIV_active"].value_counts()[1]
    inactive=training_data["HIV_active"].value_counts()[0]
    print(f"Class balance in training data after oversampling - HIV_active: {active} HIV_inactive {inactive}.")
    print(f"Number of compounds in training data after oversampling: {training_data.shape[0]}")
    print(f"Number of compounds in testing data after oversampling: {testing_data.shape[0]}")
    print(f"Number of compounds in total after preprocessing: {testing_data.shape[0] + training_data.shape[0]}")

    # Save preprocessed test and train dataset
    training_data.to_csv(path_or_buf = "data/df_train_preprocessed.csv")
    testing_data.to_csv(path_or_buf = "data/df_test_preprocessed.csv")
else:
    print("Found local preprocessed dataframe")
    training_data = pd.read_csv("data/df_train_preprocessed.csv", index_col=[0])
    testing_data = pd.read_csv("data/df_test_preprocessed.csv", index_col=[0])

# creating a list of inactive and active compounds
df_inactives_train = training_data.loc[(training_data['HIV_active'] == 0)].reset_index(drop=True)
df_actives_train = training_data.loc[(training_data['HIV_active'] == 1)].reset_index(drop=True)

df_inactives_test = testing_data.loc[(testing_data['HIV_active'] == 0)].reset_index(drop=True)
df_actives_test = testing_data.loc[(testing_data['HIV_active'] == 1)].reset_index(drop=True)

# --------MolImages--------
# creating list of RDKit mol objects for inactive and active compounds
inactivesList_train = getMolListFromDataFrame(df_inactives_train, "MolName")
activesList_train = getMolListFromDataFrame(df_actives_train, "MolName")

inactivesList_test = getMolListFromDataFrame(df_inactives_test, "MolName")
activesList_test = getMolListFromDataFrame(df_actives_test, "MolName")

# train images
produceMolImages(path="Images/MolFromSmilesImages/Train/", compoundList=inactivesList_train, HIV_activity="inactive")
produceMolImages(path="Images/MolFromSmilesImages/Train/", compoundList=activesList_train, HIV_activity="active")

# test images
produceMolImages(path="Images/MolFromSmilesImages/Test/", compoundList=inactivesList_test, HIV_activity="inactive")
produceMolImages(path="Images/MolFromSmilesImages/Test/", compoundList=activesList_test, HIV_activity="active")

# --------SmilesImages--------
# uses smiles to produce images

# train images
generateImageSMILE(path="Images/SmilesImages/Train/", compoundList=df_inactives_train, HIV_activity="inactive")
generateImageSMILE(path="Images/SmilesImages/Train/", compoundList=df_actives_train, HIV_activity="active")

# test images
generateImageSMILE(path="Images/SmilesImages/Test/", compoundList=df_inactives_test, HIV_activity="inactive")
generateImageSMILE(path="Images/SmilesImages/Test/", compoundList=df_actives_test, HIV_activity="active")

# --------SmilesColorImages--------
# uses smiles to produce images

# train images
generateImageSMILEColor(path="Images/SmilesColorImages/Train/", compoundList=df_inactives_train, HIV_activity="inactive", withChars=False)
generateImageSMILEColor(path="Images/SmilesColorImages/Train/", compoundList=df_actives_train, HIV_activity="active", withChars=False)

# test images
generateImageSMILEColor(path="Images/SmilesColorImages/Test/", compoundList=df_inactives_test, HIV_activity="inactive", withChars=False)
generateImageSMILEColor(path="Images/SmilesColorImages/Test/", compoundList=df_actives_test, HIV_activity="active", withChars=False)
