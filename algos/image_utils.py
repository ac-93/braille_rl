# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import os
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt
import pickle
from skimage.util import random_noise
import json
import itertools
import time

def plot_state(state, m):
    ''' iterates through state buffer, plotting each image.
    '''
    for i in range(m):
        img = state[:,:,i]
        plt.imshow(img, cmap='gray')
        plt.title('Frame {}'.format(i))
        plt.show()

def load_keras_model(model_dir):
    # load params
    params = load_json_obj(model_dir+'params_dict')

    # load model
    model = keras.models.load_model(model_dir+'best_model.h5')
    print(model.summary())

    return model, params

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
    ''' Converts an image to uint8 format, useful for displaying images
    '''
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = 255 * image # Now scale by 255
    return image.astype(np.uint8)

def find_nearest(array, value):
    ''' Returns value from array that is closes approximation to calue specified
        using absolute difference.
    '''
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

def permutation(lst):
    """
    Python function to return all permutations of a given list
    """
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]

    l = [] # empty list that will store current permutation
    # Iterate the input(lst) and calculate the permutation
    for i in range(len(lst)):
       m = lst[i]
       # Extract lst[i] or m from the list.  remLst is
       # remaining list
       remLst = lst[:i] + lst[i+1:]
       # Generating all permutations where m is first
       # element
       for p in permutation(remLst):
           l.append([m] + p)
    return l


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          dirname=None,
                          save_flag=False):
    """
    This function plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches((12, 12), forward=False)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', origin='upper', extent=[-0.50, len(classes)-0.5, len(classes)-0.5, -0.5], cmap=cmap)

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

    ax = plt.gca()
    offset=0.5
    ax.set_xlim(-offset, len(classes) - offset)
    ax.set_ylim(len(classes) - offset, -offset,)

    plt.tight_layout()
    plt.ylabel('Goal Key', fontsize=16, fontweight='bold')
    plt.xlabel('Key Pressed',fontsize=16, fontweight='bold')

    if save_flag:
        fig.savefig(dirname+'/cnf_mtrx.png',dpi=320, pad_inches=0.01, bbox_inches='tight', transparent=True)

    plt.show()

def process_image(image, gray=True, bbox=None, dims=None, stdiz=False, rshift=None, rzoom=None, thresh=False, add_axis=False, brightlims=None, noise_var=None):
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

    return image

def threshold_image(image):
    """Performs adaptive thresholding of a Numpy image tensor.
    """
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,-30)
    # image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,-20)
    return image


def random_image_brightness(image, brightlims):
    """Performs brightness and contrast adjustments of a Numpy image tensor.
    """
    if image.dtype != np.uint8:
        raise ValueError('This random brightness should only be applied to uint8 images on a 0-255 scale')

    a1,a2,b1,b2 = brightlims
    alpha = np.random.uniform(a1,a2)  # Simple contrast control
    beta =  np.random.randint(b1,b2)  # Simple brightness control
    new_image = np.clip(alpha*image + beta, 0, 255).astype(np.uint8)

    return new_image

def random_image_noise(image, noise_var):
    """Performs additive guasian distributed noise of a Numpy image tensor.
    """
    new_image = random_noise(image, var=noise_var)
    return new_image

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
