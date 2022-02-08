from __future__ import absolute_import
from __future__ import print_function
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Activation, Concatenate, Dot

from os import path
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
from os import path
import argparse
import tensorflow as tf
#import tensorflow_addons as tfa

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Activation, Concatenate, Dot
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import pathlib






def load_path_dataset():
    ''' Train set Left / Right'''
    L1 = pathlib.Path('data/train_set/L')
    R1 = pathlib.Path('data/train_set/R')

    ''' Test set Left / Right'''
    L2 = pathlib.Path('data/test_set/L')
    R2 = pathlib.Path('data/test_set/R')

    return L1, R1, L2, R2

L1, R1, L2, R2 = load_path_dataset()


def ssi_loss(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return   1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=sub2, filter_size=2, filter_sigma=1.5, k1=0.01, k2=0.03))

def ssi_acc(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return  tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=sub2, filter_size=2, filter_sigma=1.5, k1=0.01, k2=0.03))



## per ora questa versione è buggata, bisogna specificare i power factors, comunque è quella consigliata in questo paper:
## https://arxiv.org/pdf/1511.08861.pdf
def ssi_loss_ms(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=sub2, filter_size=2, power_factors=[(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)] ))

def ssi_acc_ms(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=sub2, filter_size=2, power_factors=[(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)] ))


def loss_smooth(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1)
    return l_edges

def acc_smooth(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1)
    return 1.0 - l_edges

#Depth-wise loss
def loss_depth_wise(y_true, y_pred):
    l_depth = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    return l_depth

def acc_depth_wise(y_true, y_pred):
    l_depth = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    return 1.0 - l_depth

def loss_final(y_true, y_pred):
    return alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred)

def acc_final(y_true, y_pred):
    return 1.0 - (alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred))

#Convex combination of the losses above
def convex_comb_loss(alpha1, alpha2, alpha3):
    def loss_final(y_true, y_pred):
        return alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred)
    return loss_final

def convex_comb_acc(alpha1, alpha2, alpha3):
    def acc_final(y_true, y_pred):
        return 1.0 - (alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred))
    return acc_final


#define personalized loss
alpha1=0.3 
alpha2=0.6 
alpha3=0.1
cc_loss =  convex_comb_loss(alpha1, alpha2, alpha3)
cc_acc  =  convex_comb_acc(alpha1, alpha2, alpha3)




autoencoder = tf.keras.models.load_model('depth_ep18.hdf5',
                                         custom_objects={'ssi_loss': ssi_loss,
                                                         'ssi_acc': ssi_acc,
                                                         'cc_acc': cc_acc,
                                                         'loss_final': loss_final,
                                                         'acc_final': acc_final })






IMG_HEIGHT = 192
IMG_WIDTH = 384

import matplotlib.pyplot as plt







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

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    if heatmap.numpy().max() > 0:
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    else:
        heatmap = tf.maximum(-heatmap, 0) / tf.math.reduce_max(-heatmap)
    return heatmap.numpy()


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
    #superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img[0])

    # Save the superimposed image
    plt.title(title)
    #    plt.colorbar()
    plt.imshow(superimposed_img)
    plt.colorbar(label="Pixel Relevance", orientation="vertical",shrink=0.9)
    plt.savefig(path_to_save + "/" + title)
    plt.clf()


import numpy as np

autoencoder.summary()
autoencoder.get_layer('functional_1').summary()
autoencoder.layers[-1].activation = None

X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")
it = 0
import os
for i in range(X_test.shape[0]):
    result_np = []
    path_to_save = "test_img_depth/X_test/%d" % it
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    img1 =X_test[i]
    plt.imshow(np.asarray(img1, dtype=np.uint8))
    plt.savefig(path_to_save + "/input")

    img1 = img1[:, :, 0].reshape([1, IMG_HEIGHT,IMG_WIDTH,1]) / 255.
    img2 = img1

    plt.imshow(Y_test[i])
    plt.colorbar(orientation='horizontal')
    plt.savefig(path_to_save + "/mask")
    plt.clf()
    autoencoder.summary()
    autoencoder.get_layer('functional_1').summary()
    heatmap = make_gradcam_heatmap([img1, img2], autoencoder.get_layer('functional_1'), 'conv2d_28')
    heatmappp = grad_cam_plus(autoencoder.get_layer('functional_1'), [img1, img2], layer_name="conv2d_28")
    result_np.append(heatmap)
    result_np.append(heatmappp)
    result_np.append(((heatmappp + heatmap) / 2.))
    result_np.append(np.absolute(heatmappp - heatmap))

    plt.imshow(np.absolute(heatmappp - heatmap), cmap="viridis")
    plt.colorbar(orientation='horizontal')
    plt.savefig(path_to_save + "/InformationContent")
    plt.clf()

    plt.imshow(heatmap, cmap="viridis")
    plt.colorbar(orientation='horizontal')
    plt.savefig(path_to_save + "/gmap")
    plt.clf()

    plt.imshow(heatmappp, cmap="viridis")
    plt.colorbar(orientation='horizontal')
    plt.savefig(path_to_save + "/gmappp")
    plt.clf()

    plt.imshow(((heatmappp + heatmap) / 2.),  cmap="viridis")
    plt.colorbar(orientation='horizontal')
    plt.savefig(path_to_save + "/InformationContentMean")
    plt.clf()
    decoded_imgs = autoencoder.predict([img1, img2]) #?
    decoded_imgs = (decoded_imgs - np.min(decoded_imgs)) / np.ptp(decoded_imgs)
    plt.imshow(decoded_imgs.squeeze(), cmap='inferno')
    plt.colorbar(orientation='horizontal')
    plt.savefig(path_to_save + "/predicted")
    plt.clf()
    save_and_display_gradcam(img1, heatmap, "Gradcam", path_to_save)
    save_and_display_gradcam(img1, heatmappp, "Gradcampp", path_to_save)
    heatmap = make_gradcam_heatmap([img1, img2], autoencoder.get_layer('functional_1'), 'up_sampling2d_3')
    heatmappp = grad_cam_plus(autoencoder.get_layer('functional_1'), [img1, img2], layer_name="up_sampling2d_3")
    result_np.append(heatmap)
    result_np.append(heatmappp)
    result_np.append(((heatmappp + heatmap) / 2.))
    result_np.append(np.absolute(heatmappp - heatmap))
    plt.imshow(np.absolute(heatmappp - heatmap), cmap="viridis")
    plt.savefig(path_to_save + "/InformationContent1")
    plt.imshow(heatmap, cmap="viridis")
    plt.savefig(path_to_save + "/gmap1")
    plt.imshow(heatmappp, cmap="viridis")
    plt.savefig(path_to_save + "/gmappp1")
    plt.imshow(((heatmappp + heatmap) / 2.),  cmap="viridis")
    plt.savefig(path_to_save + "/InformationContentMean1")
    save_and_display_gradcam(img1, heatmap, "Gradcam1", path_to_save)
    save_and_display_gradcam(img1, heatmappp, "Gradcampp1", path_to_save)
    result_np.append(decoded_imgs.squeeze())
    result_np.append(Y_test[i])
    result_np = np.asarray(result_np)
    np.save(path_to_save + "/result.npy", result_np)

    it += 1

