import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Descriptors import MolWt,NumRotatableBonds,HeavyAtomCount
from collections import Counter
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle
from rdkit.Chem import AllChem

#from tdc.single_pred import HTS
#from tqdm import tqdm
#tqdm.pandas()

def load_dataset():
    """
    Loads the HIV dataset 
    :return df: HIV dataset with renamed columns 
    """
    data = HTS(name = 'HIV')
    df = data.get_data()
    #df.drop(['Drug'], axis=1, inplace=True)
    df.rename(columns={'Drug_ID':'MolName','Drug':'SMILES','Y':'HIV_active'},inplace=True)
    return df

def commonAtoms(smi):
    """
    Removes moleculs with uncommon atoms
    :param smi: molecule to be checked  
    :return df: True if common atom, otherwise false 
    """
    commonAtomNumbers = [1,6,7,8,9,15,16,17,35,53]
    mol = Chem.MolFromSmiles(smi)
    res = True
    if mol:
        for atm in mol.GetAtoms():
            if not atm.GetAtomicNum() in commonAtomNumbers:
                res = False
                break
    else:
        res = False
    return res

def neutralize_atoms(smi):
    """
    neutralize 
    :param smi: molecule to be checked
    :return smi: neutralized atom 
    """
    #RDLogger.DisableLog('rdApp*')                                                                                                                                                       
    mol = Chem.MolFromSmiles(smi)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    #RDLogger.DisableLog('rdApp*')                                                                                                                                                       
    smi = Chem.MolToSmiles(mol)
    return smi

def stripSalts(smi):
    """
    ...
    :param smi: molecule to be checked
    :return : return max value in smi_longest
    """
    smi_longest = smi.split(".")
    return max(smi_longest, key=len)

def calc_3_descriptors(smi):
    """
    ...
    :param smi: molecule to be checked 
    :return res: list of molecular descriptors
    """
    mol = Chem.MolFromSmiles(smi)
    if mol:
        mw,rotors,hvys = [x(mol) for x in [MolWt, NumRotatableBonds, HeavyAtomCount]]
        res = [mw,rotors,hvys]
    else:
        res = [None] * 3
    return res

def oversample(df, feature):
    """
    Oversampling the minority class so the imbalance ration becomes equal
    :param df: dataframe to check
    :param feature: feature to check for imbalance
    :return df_oversampled: oversampeled dataset
    """
    # Count how many samples for each class is present
    counts = Counter(df[feature])

    # Determine max class and count
    maxclass, maxcount = Counter(df[feature]).most_common(1)[0]

    # Construct separate list of each class to match max class and concat to single list
    index_lists = []
    for key in counts.keys():
        tmp_list = [i for i in range(len(df[feature])) if df[feature][i] == key]
        index_lists.append(tmp_list)
        # Oversample non-max class until max count is reached
        if len(tmp_list) < maxcount:
            index_lists.append(np.random.choice(tmp_list, size=maxcount-len(tmp_list)))#, replace=True))
    index_list = np.concatenate(index_lists)
    np.random.shuffle(index_list)

    # Construct a new df with index_list
    df_oversampled = pd.DataFrame(columns=["MolName", "SMILES", "HIV_active"])
    for index in index_list:
        df_oversampled = df_oversampled.append(df.loc[index])
    
    # reset index
    df_oversampled = df_oversampled.reset_index(drop=True)
    
    return df_oversampled

def tensorDataPrep(loadPath, savePath, testOrTrain):
    data = []
    # creata train and test data
    for img in os.listdir(loadPath):
        img_array = cv2.imread(os.path.join(loadPath + img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        if "inactive" in img:
            label=0
        else:
            label=1
        data.append([img_array, label])

    data = pd.DataFrame(data)

    pickle_out = open(savePath + "/" + testOrTrain + "Data" + ".pickle","wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

    return data # X, y

def chemcepterize_mol(mol, embed=20.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims,4))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            vect[ idx , idy ,0] = bondorder
    #Atom Layers
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 1] = atom.GetAtomicNum()
            #Gasteiger Charges
            charge = atom.GetProp("_GasteigerCharge")
            vect[ idx , idy, 3] = charge
            #Hybridization
            hyptype = atom.GetHybridization().real
            vect[ idx , idy, 2] = hyptype

    return vect

