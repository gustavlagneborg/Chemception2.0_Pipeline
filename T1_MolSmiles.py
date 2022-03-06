import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random

DirTrain = "Images/MolFromSmilesImages/Train/"
DirTest = "Images/MolFromSmilesImages/Test/"

# creata train and test data
for img in os.listdir(DirTrain)[:10]:
    img_array = cv2.imread(os.path.join(DirTrain + img))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    print(img_array.shape)
    plt.imshow(img_array)
    plt.show()

# shuffle the data
random.shuffle()

# 5-fold CV