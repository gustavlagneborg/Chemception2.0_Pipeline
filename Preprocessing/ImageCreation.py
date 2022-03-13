from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem
from PIL import Image
from PIL import ImageDraw
import numpy as np
#import matplotlib.pyplot as plt
from rdkit.Chem import AllChem


# --------MolImages--------

def getMolListFromDataFrame(df, prop_name):
    """
    Creating a mol object in a df containing smile strings
    :param df: dataframe with compounds
    :param prop_name: name on column
    :return molObjList: list of mol objects 
    """
    molObjList = []    
    for ind, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        for prop_name in [s for s in row.index if s !="SMILES"]: # all other cols except smiles 
            mol.SetProp(prop_name,str(row[prop_name])) # only str allowed in mol props
        molObjList.append(mol)
    return molObjList

def produceMolImages(path, compoundList, HIV_activity):
    """
    Producing molecular images from RDKit
    :param path: the file location to save the images
    :param compoundList: list of compounds
    :param HIV_activity: active or inactive compounds, used to name to image correctly
    """

    # Counter to make each image name unique since oversampling has created copies of different samples
    counter = 0

    for active in compoundList:
        molname = active.GetProp("MolName")
        molname = molname.replace(" ", "_")
        filename = path + molname + "_" + HIV_activity + str(counter) + ".png"
        counter += 1
        Chem.Draw.MolToFile(active, filename, size=(200,150))#, kekulize=True, wedgeBonds=True, imageType=None, fitImage=False, options=None)
    
    print(f"Done producing molImages {HIV_activity} compounds for path: {path}. {counter} images was created")
    

# Smiles Image setup
# Colors from https://sciencenotes.org/molecule-atom-colors-cpk-colors/
colors = {
    'H'  : "#FFFFFF",
    'C'  : "#909090",
    'O'  : "#FF0D0D",
    'N'  : "#3050F8",
    'S'  : "#FFFF30",
    'P'  : "#FF8000",
    'B'  : "#FFB5B5",
    'F'  : "#90E050",
    'K'  : "#8F40D4",
    'I'  : "#940094",
    'Na' : "#AB5CF2",
    'Cl' : "#1FF01F",
    'Br' : "#A62929",
    '_'  : "#000000", # Color for bonds, connections and brackets
         }

colors = {k : np.array([int(v[1:3],16) / 255.,
                        int(v[3:5],16) / 255.,
                        int(v[5:7],16) / 255.])
             for k,v in colors.items()
         }




look_ahead = {
    'C' : 'l',
    'N' : 'a',
    'B' : 'r'
}


skip = ["@", "[", "]"]


def colorSMILES(smi):
    max_nested = 10
    max_connections = 10
    width = 12 
    height = 18
    off = width // 2
    
    upper = max_nested
    mid = upper + height // 2 
    lower = upper + height
    max_len = 50 # Number of atoms and "non-singel" bonds 
    lvl = 0
    active_con = [-1] * max_connections
    im_arr = np.ones((height+max_connections+max_nested,
                      width * max_len, 3))
    
    
    
    parsed = False
    pos = 0
    
    startP = [False] * max_nested
    
    for i,c in enumerate(smi):
        
        
        if c in skip:
            continue
        
        if parsed:
            parsed = False
            continue
        
        full = c.isupper()
        
        if c == "(":
            startP[lvl] = True
            lvl += 1
            continue
        elif c == ")":
            lvl -= 1
            continue

            
        if c.isnumeric():
            n = int(c)-1
            ended_con = active_con[n]
            if active_con[n] < 0:
                for k in range(max_nested):
                    if not k in active_con:
                        break
                active_con[n] = k
                im_arr[lower+k:lower+1+k,pos-width:pos] = colors["_"]
            else:
                active_con[n] = -1
            continue
            
        if c in look_ahead and i+1 < len(smi) and (
                smi[i+1] and look_ahead[c] == smi[i+1]):
            c += smi[i+1]
            parsed = True
        else:
            c = c.upper()

        if c in colors:
            col  = colors[c]
            u = upper if full else mid
            im_arr[u:lower,pos:pos+width] = col            
            
        else:
            col = colors["_"]
            
            if c == "-" or c == "#":
                im_arr[mid,pos:pos+width] = col  
            if c == "=" or c == "#":
                im_arr[[mid + x for x in [-3,3]],
                       pos:pos+width] = col
                
        
        # Parentheses
        for i in range(lvl):            
            ps = pos + (off if startP[i] else 0)
            pe = pos + width
            im_arr[upper-i-1:upper-i,ps:pe] = colors["_"]
            startP[i] = False

            
        # Connections
        for i in range(len(active_con)):            
            if active_con[i] >= 0:
                k = active_con[i]
                im_arr[lower+k:lower+1+k,pos:pos+width] = colors["_"]
                
        
        
        
        pos += width

    startP = []
    stopP = []

    return im_arr




def SMILESToImage(smi, background=None):
    
    char_width = 6
    height = 17
    num_chars = 100
    width = char_width * num_chars
    
    if background is None:
        img = Image.new("RGB", (width, height), (255, 255, 255))
    else:
        img = background
    
    
    draw = ImageDraw.Draw(img)
    w, h = draw.textsize(smi)
    
    p = 0
    saved = ""
    
    for i,c in enumerate(smi):
        
        if c in skip:
            continue
        
        if c.isnumeric() or c in ["(", ")"]:
            continue
            
        if c in look_ahead and i+1 < len(smi) and (
                smi[i+1] and look_ahead[c] == smi[i+1]):
            saved += c
            continue

        
        c = saved + c
        saved = ""
        o = 1 if len(c) == 1 else 0
        
        draw.text((p + o,h/2+5), c, fill="black")
        p += 2*char_width
            
    return img


# --------SmilesImages--------
def generateImageSMILE(path, compoundList, HIV_activity):

    # image setup
    char_width = 6
    height = 17
    num_chars = 100
    width = char_width * num_chars
    
    # Counter to make each image name unique since oversampling has created copies of different samples
    counter = 0

    for index, compound in compoundList.iterrows():
        filename = path + compound["MolName"].replace(" ", "_") + "_" + HIV_activity + str(counter) + ".png"
        counter += 1

        img = Image.new("RGB", (width, height), (255, 255, 255))
        
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(compound["SMILES"])

        draw.text((0,h/2), compound["SMILES"], fill="black")

        img.save(fp=filename)

    print(f"Done producing smilesImages {HIV_activity} compounds for path: {path}. {counter} images was created")

# --------SmilesColorImages--------
def generateImageSMILEColor(path, compoundList, HIV_activity, withChars=True):

    # Counter to make each image name unique since oversampling has created copies of different samples
    counter = 0

    for index, compound in compoundList.iterrows():
        smi = str(compound["SMILES"])
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol)
       
        filename = path + compound["MolName"].replace(" ", "_") + "_" + HIV_activity + str(counter) + ".png"
        counter += 1
    
        bg = Image.fromarray((colorSMILES(smi) * 255).astype("uint8"))    
        if withChars:
            SMILESToImage(smi, bg).save(fp=filename)
        else:
            bg.save(fp=filename)
    
    print(f"Done producing smilesColorImages {HIV_activity} compounds for path: {path}. {counter} images was created")


# --------ChemOriginal Images--------
def chemcepterize_mol(path, compoundList, HIV_activity, embed=20.0, res=0.5):
    
    # Counter to make each image name unique since oversampling has created copies of different samples
    counter = 0

    for index, compound in compoundList.iterrows()[:10]:
        filename = path + compound["MolName"].replace(" ", "_") + "_" + HIV_activity + str(counter) + ".png"

        dims = int(embed*2/res)
        cmol = Chem.Mol(compound.ToBinary())
        cmol.ComputeGasteigerCharges()
        AllChem.Compute2DCoords(cmol)
        coords = cmol.GetConformer(0).GetPositions()
        vect = np.zeros((dims,dims,4))
        #Bonds first
        for i,bond in enumerate(compound.GetBonds()):
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
    
        vect.save(fp=filename)
    
    print(f"Done producing ChemOriginal Images {HIV_activity} compounds for path: {path}. {counter} images was created")
