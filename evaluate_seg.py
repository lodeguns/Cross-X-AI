import os
import sys
import json
import datetime
import numpy as np
#import skimage.draw
import cv2
import matplotlib.pyplot as plt
#from pycocotools.coco import COCO
from PIL import Image

ROOT_DIR = os.path.abspath("./")


X_test = np.load(os.path.join(ROOT_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(ROOT_DIR, "Y_test.npy"))
#To install the pretrained segmentation models
#pip install git+https://github.com/qubvel/segmentation_models


import segmentation_models as sm
from tensorflow import keras
import tensorflow as tf
sm.set_framework('tf.keras')
BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)
keras.backend.clear_session()

LR = 0.0001
checkpoint_filepath = ROOT_DIR
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True)

# preprocess input

X_test_ = X_test / 255.
# define model
'''
model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
model.compile(
    keras.optimizers.Adam(LR),
    loss='binary_crossentropy',
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
)

# fit model

history = model.fit(
   x=X_train_,
   y=Y_train,
   batch_size=8,
   epochs=40,
   validation_data=(X_val_, Y_val),
   callbacks=[model_checkpoint_callback]
)

#Save training metrics
import json


json.dump( history.history, open( os.path.join(ROOT_DIR, "metrics.json"), 'w' ) )


data = json.load( open( os.path.join(ROOT_DIR, "metrics.json") ) )
'''
import tensorflow as tf
from tensorflow.keras import Model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    import numpy as np
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:

        last_conv_layer_output, preds = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        class_channel = preds

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print("------------", grads)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    '''
    heatmap = last_conv_layer_output * pooled_grads
    heatmap = np.sum(tf.maximum(heatmap,0),axis=-1)
    print(heatmap.max())
    '''
    heatmap = tf.squeeze(heatmap)
    '''
    numer = heatmap - np.min(heatmap)
    denom = (tf.math.reduce_max(heatmap) - tf.math.reduce_min(heatmap)) + 1e-8
    heatmap = numer / denom
    heatmap = (heatmap.numpy() * 255).astype("uint8")
    #heatmap = tf.math.abs(heatmap) / tf.math.reduce_max(heatmap)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    '''
    if heatmap.numpy().max() > 0:
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    else:
        heatmap = tf.maximum(-heatmap, 0) / tf.math.reduce_max(-heatmap)
    heatmap = np.uint8(255 * heatmap)

    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def grad_cam_plus(model, img,
                  layer_name, label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.
    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.
    Return:
        A heatmap ndarray(without color).
    """
    import numpy as np
    img_tensor = img

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                '''
                if category_id == None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                '''
                output = predictions
                conv_first_grad = gtape3.gradient(output, conv_output)
                print(conv_first_grad.numpy().min())
                print(conv_first_grad.numpy().max())

            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            print(conv_second_grad)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))
    grad_CAM_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

    heatmap = np.maximum(grad_CAM_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    '''
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    heatmap = grad_CAM_map
    numer = heatmap - np.min(heatmap)
    denom = (tf.math.reduce_max(heatmap) - tf.math.reduce_min(heatmap)) + 1e-8
    heatmap = numer / denom
    heatmap = (heatmap.numpy() * 255).astype("uint8")
    '''
    heatmap = np.uint8(255 * heatmap)

    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def save_and_display_gradcam(img, heatmap, title, path_to_save, alpha=0.4):
    # Load the original image
    from tensorflow import keras
    import numpy as np
    import matplotlib.cm as cm
    import cv2
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("viridis")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((img.shape[2], img.shape[1]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    print(img.shape)
    # Superimpose the heatmap on original image
    img = cv2.cvtColor(img.squeeze(), cv2.COLOR_BGR2RGB)

    superimposed_img = np.asarray(jet_heatmap * alpha + (img * 255.), dtype=np.uint8)
    # superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img[0])

    # Save the superimposed image
    plt.title(title)
    plt.imshow(superimposed_img)
    plt.colorbar(label="Pixel Relevance", orientation="vertical", shrink=0.9)
    plt.savefig(path_to_save + "/" + title)
    plt.clf()




from matplotlib import pyplot as plt
import segmentation_models as sm
from tensorflow import keras

sm.set_framework('tf.keras')
BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)
keras.backend.clear_session()
LR = 0.0001
model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
model = keras.models.load_model(os.path.join(ROOT_DIR), compile=False)
#X_test = X_test_
for ind in range(X_test.shape[0]):
    path_to_save = os.path.join(ROOT_DIR, "X_test_unet", "%d" % ind)
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    heatmap = make_gradcam_heatmap(np.expand_dims(X_test[ind]/255., axis=0), model, 'final_conv')
    heatmappp = grad_cam_plus(model, np.expand_dims(X_test[ind]/255., axis=0), layer_name="final_conv")
    pred = model.predict(np.expand_dims(X_test[ind]/255., axis=0))
    save_and_display_gradcam(np.expand_dims(X_test[ind]/255., axis=0), heatmap, "GRADCAM", path_to_save)
    save_and_display_gradcam(np.expand_dims(X_test[ind]/255., axis=0), heatmappp, "GRADCAMpp", path_to_save)
    plt.imshow(pred.squeeze())
    plt.title("Predizione")
    plt.savefig(path_to_save + "/prediction")

    plt.imshow(np.asarray(X_test[ind], dtype=np.uint8))
    plt.title("Input")
    plt.savefig(path_to_save + "/input")

    plt.imshow(Y_test[ind])
    plt.title("GT")
    plt.savefig(path_to_save + "/groundTruth")

    plt.imshow(heatmap)
    plt.title("gradcam")
    plt.savefig(path_to_save + "/gmap")

    plt.imshow(heatmappp)
    plt.title("gradcam++")
    plt.savefig(path_to_save + "/gmappp")
    np.save(os.path.join(path_to_save, "input.npy"), X_test[ind])
    np.save(os.path.join(path_to_save, "pred.npy"), pred.squeeze())
    np.save(os.path.join(path_to_save, "gt.npy"), Y_test[ind])
    np.save(os.path.join(path_to_save, "heatmap.npy"), heatmap)
    np.save(os.path.join(path_to_save, "heatmappp.npy"), heatmappp)




