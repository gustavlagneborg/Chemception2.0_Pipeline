import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import numpy as np
import tensorflow as tf
from MachineLearning.evaluation import *
from Preprocessing.DataPrep import *
from Preprocessing.ImageCreation import *
from PIL import Image
from lime import lime_image
import time
from skimage.segmentation import mark_boundaries
from GradCam import GradCAM

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
df_test["imagesSmiles"] = df_test["SMILES"].apply(generateImageSMILEColorTestData, args=[True])
df_test["tensor"] = df_test["images"].apply(np.array)
df_test["tensorSmiles"] = df_test["imagesSmiles"].apply(np.array)

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

# lime config
explainer = lime_image.LimeImageExplainer(random_state=random_state)

# Grad-Cam config
last_conv_layer_name = "conv2d_62"

# produce explanations for active compounds
for index, row in df_correct_active[:3].iterrows():
    fig, axs = plt.subplots(6)
    fig.suptitle("Explanation for HIV active compound: " + row["MolName"])
    fig.tight_layout()

    # plot original image
    axs[0].imshow(row["tensor"])
    axs[0].set_title("Original image")
    axs[0].axes.xaxis.set_visible(False)
    axs[0].axes.yaxis.set_visible(False)

    axs[1].imshow(row["tensorSmiles"])
    axs[1].set_title("Original image with smiles")
    axs[1].axes.xaxis.set_visible(False)
    axs[1].axes.yaxis.set_visible(False)

    explanation = explainer.explain_instance(row["tensor"],
                                             model.predict,
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=True)

    # plot lime exlanation
    image_explanations = mark_boundaries(temp.astype(np.uint8), mask)

    axs[2].imshow(image_explanations)
    axs[2].set_title("Lime Explanation")
    axs[2].axes.xaxis.set_visible(False)
    axs[2].axes.yaxis.set_visible(False)

    # Lime heatmap
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    mappable = axs[4].imshow(heatmap, cmap='viridis', vmin=-heatmap.max(), vmax=heatmap.max())
    #plt.colorbar(mappable, ax=axs[3])
    axs[4].set_title("Lime Heatmap")
    axs[4].axes.xaxis.set_visible(False)
    axs[4].axes.yaxis.set_visible(False)

    # plot gradcam explanation
    image = cv2.resize(row["tensor"], (600, 38))
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)

    icam = GradCAM(model, row["probability"], last_conv_layer_name)
    heatmap = icam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (600, 38))
    axs[5].set_title("Grad-Cam Heatmap")
    axs[5].imshow(heatmap)
    axs[5].axes.yaxis.set_visible(False)
    axs[5].axes.xaxis.set_visible(False)

    image = cv2.resize(row["tensor"], (600, 38))
    (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

    axs[3].set_title("Grad-Cam Explanation")
    axs[3].imshow(output)
    axs[3].axes.yaxis.set_visible(False)
    axs[3].axes.xaxis.set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    fig.savefig("active_explanation_{}".format(row["MolName"]), dpi=600)
    break

# produce explanations for active compounds
for index, row in df_correct_inactive[:3].iterrows():
    fig, axs = plt.subplots(6)
    fig.suptitle("Explanation for HIV inactive compound: " + row["MolName"])
    fig.tight_layout()

    # plot original image
    axs[0].imshow(row["tensor"])
    axs[0].set_title("Original image")
    axs[0].axes.xaxis.set_visible(False)
    axs[0].axes.yaxis.set_visible(False)

    axs[1].imshow(row["tensorSmiles"])
    axs[1].set_title("Original image with smiles")
    axs[1].axes.xaxis.set_visible(False)
    axs[1].axes.yaxis.set_visible(False)

    explanation = explainer.explain_instance(row["tensor"],
                                             model.predict,
                                             top_labels=1,
                                             hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                num_features=5,
                                                hide_rest=True)

    # plot lime explanation
    image_explanations = mark_boundaries(temp.astype(np.uint8), mask)

    axs[2].imshow(image_explanations)
    axs[2].set_title("Lime Explanation")
    axs[2].axes.xaxis.set_visible(False)
    axs[2].axes.yaxis.set_visible(False)

    # heatmap
    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    mappable = axs[4].imshow(heatmap, cmap='viridis', vmin=-heatmap.max(), vmax=heatmap.max())
    #plt.colorbar(mappable, ax=axs[3])
    axs[4].set_title("Lime Heatmap")
    axs[4].axes.xaxis.set_visible(False)
    axs[4].axes.yaxis.set_visible(False)

    # plot gradcam explanation
    image = cv2.resize(row["tensor"], (600, 38))
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)

    icam = GradCAM(model, row["probability"], last_conv_layer_name)
    heatmap = icam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (600, 38))
    axs[5].set_title("Grad-Cam Heatmap")
    axs[5].imshow(heatmap)
    axs[5].axes.xaxis.set_visible(False)
    axs[5].axes.yaxis.set_visible(False)

    image = cv2.resize(row["tensor"], (600, 38))
    (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

    axs[3].set_title("Grad-Cam Explanation")
    axs[3].imshow(output)
    axs[3].axes.xaxis.set_visible(False)
    axs[3].axes.yaxis.set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    fig.savefig("inactive_explanation_{}".format(row["MolName"]), dpi=600)
    break

plt.show()
