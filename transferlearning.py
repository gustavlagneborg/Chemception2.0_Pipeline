from Preprocessing.DataPrep import *
from Preprocessing.ImageCreation import *
import os.path

random_state = 125

train_file_exists = os.path.exists('Data/df_train_preprocessed.csv')
test_file_exists = os.path.exists('Data/df_test_preprocessed.csv')

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

    # Save preprocessed test and train dataset
    training_data.to_csv(path_or_buf ="HivData/df_train_preprocessed.csv")
    testing_data.to_csv(path_or_buf ="HivData/df_test_preprocessed.csv")
else:
    print("Found local preprocessed dataframe")
    training_data = pd.read_csv("HivData/df_train_preprocessed.csv", index_col=[0]).reset_index(drop=True)
    testing_data = pd.read_csv("HivData/df_test_preprocessed.csv", index_col=[0]).reset_index(drop=True)

# creating a list of inactive and active compounds
df_inactives_train = training_data.loc[(training_data['HIV_active'] == 0)].reset_index(drop=True)
df_actives_train = training_data.loc[(training_data['HIV_active'] == 1)].reset_index(drop=True)

df_inactives_test = testing_data.loc[(testing_data['HIV_active'] == 0)].reset_index(drop=True)
df_actives_test = testing_data.loc[(testing_data['HIV_active'] == 1)].reset_index(drop=True)