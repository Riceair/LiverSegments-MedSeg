import tensorflow as tf
import numpy as np

def crop_and_resize(img):
    img_shape = img.shape
    crop_scale = np.random.uniform(low=0.5, high=1, size=1) # 0.5-1
    crop_size = int(img_shape[0]*crop_scale) # crop size
    crop_img = tf.image.random_crop(img, (crop_size, crop_size, 1)) # crop image
    crop_img = tf.image.resize(crop_img, img_shape[:-1]) # resize back to origin size
    return crop_img

def rotation(img):
    rotate_img = tf.keras.preprocessing.image.random_rotation(img, 90, row_axis=0, col_axis=1,channel_axis=2)
    return rotate_img

def random_apply(func, x, p):
    if tf.random.uniform([], minval=0, maxval=1) < p:
        return func(x)
    else:
        return x