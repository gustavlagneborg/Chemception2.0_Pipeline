import numpy as np
import tensorflow as tf
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *
from Preprocessing.ImageCreation import *
from PIL import Image
from lime import lime_image
import time
from skimage.segmentation import mark_boundaries


print(tf.__version__)

# configuration
random_state = 125
tf.random.set_seed(
    random_state
)

loadPath = "../SavedModels/BestModel/"
DirTestImg = "../HIVImages/SmilesColorImages/Test/"
DirTensorArray = "../HivData/SmilesColorArray/"
DirTestDf = "../HivData/"

# Model config
batch_size = 32
nb_epoch = 200
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

# Load test data that holds the truth
df_test = pd.read_csv(DirTestDf + "df_test_preprocessed.csv").drop(['Unnamed: 0'], axis=1)

# preprocessing test data
df_test["images"] = df_test["SMILES"].apply(generateImageSMILEColorTestData, args=[False])
df_test["tensor"] = df_test["images"].apply(np.array)

X_test = df_test["tensor"].values
X_test = np.asarray(X_test.tolist())
y_test = df_test["HIV_active"].values
y_test = tf.one_hot(y_test, depth=2)
y_test = tf.cast(y_test, tf.int32)

input_shape = X_test.shape[1:]

# Load pretrained model
filecp = loadPath + "_bestweights_trial_" + ".hdf5"
model, submodel = cs_setup_cnn(params, inshape=input_shape, classes=2, lr=0.0001)
model.load_weights(filecp, by_name=True)

print(model.summary())

# print test HivData shape and input shape
print("X_test HivData shape: " + str(X_test.shape))
print("y_test HivData shape: " + str(y_test.shape) + "\n")
print("Model input shape: " + str(input_shape) + "\n")

# predict test data
y_pred = model.predict(X_test, batch_size=batch_size)

# add predictions to df_test
df_test["probability"] = y_pred.max(1).tolist()
df_test["rounded prediction"] = np.argmax(y_pred, axis=1).tolist()

# create a df with only correct predicitions
df_correct = df_test[df_test["HIV_active"] == df_test["rounded prediction"]]
df_correct_active = df_correct[df_correct["HIV_active"] == 1].sort_values(by=["probability"], ascending=False)
df_correct_inactive = df_correct[df_correct["HIV_active"] == 0].sort_values(by=["probability"], ascending=False)

# Collect top 3 inactive and active molecules that are correctly predicted
top_active_image = df_correct_active.iloc[0]
top_inactive_image = df_correct_inactive.iloc[0]

# produce explanations
explainer = lime_image.LimeImageExplainer()

def _predict_fn(images):
    return model.predict(images)

tmp = time.time()
explanation = explainer.explain_instance(top_active_image["tensor"], _predict_fn, top_labels=1, hide_color=0, num_samples=1000)
print(time.time() - tmp)

temp, mask = explanation.get_image_and_mask(top_active_image["HIV_active"], positive_only=True, num_features=5, hide_rest=True)
#image_explanations = mark_boundaries(temp / 2 + 0.5, mask)

v=top_active_image["tensor"]
plt.imshow(v[:, :, :3])
plt.show()

image_explanations = mark_boundaries(temp, mask)
plt.imshow(image_explanations)
#plt.imshow((image_explanations*2)/255)
plt.show()