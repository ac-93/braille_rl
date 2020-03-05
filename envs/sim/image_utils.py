# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import scipy
import cv2
import matplotlib.pyplot as plt
import pickle
from skimage.util import random_noise
import json
import itertools
from sklearn.metrics import confusion_matrix
import time

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import keras.backend as K

# Save the dictionaries
def save_json_obj(obj, name):
    with open(name + '.json', 'w') as fp:
        json.dump(obj, fp)

def load_json_obj(name):
    with open(name + '.json', 'r') as fp:
        return json.load(fp)

def print_sorted_dict(dict):
    for key in sorted(iter(dict.keys())):
        print('{}:{}'.format(key, dict[key]) )

def convert_image_uint8(image):
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = 255 * image # Now scale by 255
    return image.astype(np.uint8)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def pixel_diff_norm(frames):
    ''' Computes the mean pixel difference between the first frame and the
        remaining frames in a Numpy array of frames.
    '''
    n, h, w, c = frames.shape
    pdn = [cv2.norm(frames[i], frames[0], cv2.NORM_L1) / (h * w) for i in range(1, n)]
    return np.array(pdn)

def load_video_frames(filename):
    ''' Loads frames from specified video 'filename' and returns them as a
        Numpy array.
    '''
    frames = []
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        captured, frame = vc.read()
        if captured:
            frames.append(frame)
        while captured:
            captured, frame = vc.read()
            if captured:
                frames.append(frame)
        vc.release()
    return np.array(frames)

def detect_pins(frame, min_threshold, max_threshold, min_area, max_area,
                min_circularity, min_convexity, min_inertia_ratio):
    """ Detect pins using OpenCV blob detector and specified parameters.
    """
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByConvexity = True
    params.minConvexity = min_convexity
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia_ratio

    detector = cv2.SimpleBlobDetector_create(params)

    pins = []
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(frame)
    pins.append(np.array([k.pt for k in keypoints]))

    return pins, keypoints

def process_image(image, gray=True, bbox=None, dims=None, stdiz=False, normlz=False, rshift=None, rzoom=None, thresh=False, add_axis=False, brightlims=None, noise_var=None):
    ''' Process raw image (e.g., before applying to neural network).
    '''
    if gray:
        # Convert to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Add channel axis
        image = image[..., np.newaxis]

    if bbox is not None:
        # Crop to specified bounding box
        x0, y0, x1, y1 = bbox
        image = image[y0:y1, x0:x1]

    if dims is not None:
        # Resize to specified dims
        image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)

        # Add channel axis
        image = image[..., np.newaxis]

    if add_axis:
        # Add channel axis
        image = image[..., np.newaxis]

    if rshift is not None:
        # Apply random shift to image
        wrg, hrg = rshift
        image = random_shift_image(image, wrg, hrg)

    if rzoom is not None:
        # Apply random zoom to image
        image = random_zoom_image(image,rzoom)

    if thresh:
        # Use adaptive thresholding to create binary image
        image = threshold_image(image)
        image = image[..., np.newaxis]

    if brightlims is not None:
        # Add random brightness/contrast variation to the image
        image = random_image_brightness(image, brightlims)

    if noise_var is not None:
        # Add random noise to the image
        image = random_image_noise(image, noise_var)

    if stdiz:
        # Convert to float and standardise on a per frame basis
        # position of this is important
        image = per_image_standardisation(image.astype(np.float32))

    if normlz:
        # Convert to float and standardise on a per frame basis
        # position of this is important
        image = image.astype(np.float32) / 255.0

    return image

def threshold_image(image):
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,-30)
    # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,-20)
    return image

# Change brightness levels
def random_image_brightness(image, brightlims):

    if image.dtype != np.uint8:
        raise ValueError('This random brightness should only be applied to uint8 images on a 0-255 scale')

    a1,a2,b1,b2 = brightlims
    alpha = np.random.uniform(a1,a2)  # Simple contrast control
    beta =  np.random.randint(b1,b2)  # Simple brightness control
    new_image = np.clip(alpha*image + beta, 0, 255).astype(np.uint8)

    return new_image

def random_image_noise(image, noise_var):
    new_image = random_noise(image, var=noise_var)
    return new_image

def per_image_standardisation(image):
    mean = np.mean(image, axis=(0,1), keepdims=True)
    std = np.sqrt(((image - mean)**2).mean(axis=(0,1), keepdims=True))
    t_image = (image - mean) / std
    return t_image

def random_shift_image(x, wrg, hrg, fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    """
    h, w = x.shape[0], x.shape[1]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, fill_mode=fill_mode, cval=cval)
    return x


def random_zoom_image(x, zoom_range, fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: %s' % (zoom_range,))

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    x = apply_affine_transform(x, zx=zx, zy=zy, fill_mode=fill_mode, cval=cval)
    return x


def apply_affine_transform(x, theta=0, tx=0, ty=0, zx=1, zy=1,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[0], x.shape[1]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, 2, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, 3)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


## ============================================================================

def load_model(model_dir):
    # load params
    params = load_json_obj(model_dir+'params_dict')

    # load model
    model = keras.models.load_model(model_dir+'best_model.h5')
    print(model.summary())

    return model, params

def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

## ============================================================================


class LearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print('Learning Rate: ', float(K.get_value(self.model.optimizer.lr)))

class StopAtAccuracy(keras.callbacks.Callback):

    def __init__(self, target_acc=1.0):
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):

        temp_val_acc = logs.get('val_categorical_accuracy')
        if temp_val_acc >= self.target_acc:
            print('Target Accuracy ({}) Reached, Stopping Early'.format(self.target_acc))
            self.stopped_epoch = epoch
            self.model.stop_training = True


## ============================================================================

def create_confusion_matrix(model, validation_generator, params):

    len_val = validation_generator.__len__()

    predictions_array  = np.zeros([len_val*params['batch_size']])
    ground_truth_array = np.zeros([len_val*params['batch_size']])
    counter = 0
    for val_batch in validation_generator:

        val_images, val_labels  = val_batch

        predictions = model.predict(val_images)

        mapped_predictions = np.argmax(predictions, axis=1)
        mapped_labels = np.argmax(val_labels, axis=1)

        # print('Predictions: ', mapped_predictions)
        # print('Labels:      ', mapped_labels)

        for i in range(len(mapped_predictions)):
            prediction   = mapped_predictions[i]
            ground_truth = mapped_labels[i]

            predictions_array[counter]  = prediction
            ground_truth_array[counter] = ground_truth

            counter += 1

    cnf_matrix = confusion_matrix(ground_truth_array, predictions_array)

    return cnf_matrix, predictions_array, ground_truth_array


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          dirname=None,
                          save_flag=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches((12, 12), forward=False)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted label',fontsize=16, fontweight='bold')

    if save_flag:
        fig.savefig(dirname+'/cnf_mtrx.png',dpi=320, pad_inches=0.01, bbox_inches='tight', transparent=True)

    plt.show()

def plot_training(history, dirname, save_flag=False):

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches((6, 6), forward=False)

    #  "Accuracy"
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if save_flag:
        fig.savefig(dirname+'/acc.png',dpi=320, pad_inches=0.01, bbox_inches='tight')
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    if save_flag:
        fig.savefig(dirname+'/loss.png',dpi=320, pad_inches=0.01, bbox_inches='tight')

    plt.show()


def plot_conv_weights(w, layer_num=0, save_flag=False, filename=''):

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    x_dim = 8
    y_dim = num_filters/x_dim

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(int(y_dim), int(x_dim))

    title = 'Layer {} Kernels'.format(layer_num)
    fig.suptitle(title, fontsize=14)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, 0, i]

            # Plot image.
            im = ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic', aspect='auto')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', 'box')

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.11, 0.02, 0.77])
    fig.colorbar(im, cax=cbar_ax)

    # Ensure the plot is shown correctly with multiple plots
    if save_flag:
        filename = '_{}'.format(layer_num)
        plt.savefig(filename, dpi=320, pad_inches=0.01, bbox_inches='tight')

    plt.show()



def plot_conv_layer(values, layer_num, save_flag=False, filename=''):

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    x_dim = 8
    y_dim = num_filters/x_dim

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(int(y_dim), int(x_dim))

    title = 'Layer {} Feature Map'.format(layer_num)
    fig.suptitle(title, fontsize=14)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters

        if i<num_filters:

            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary_r')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    if save_flag:
        filename = '_{}'.format(layer_num)
        plt.savefig(filename, dpi=320, pad_inches=0.01, bbox_inches='tight')

    plt.show()
