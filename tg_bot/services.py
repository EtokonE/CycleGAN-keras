from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.engine.topology import Layer
from keras.engine import InputSpec
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import PIL

class ReflectionPadding2D(Layer):
    def __init__(self, **kwargs):
        self.padding = tuple((1,1))
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__()

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')


def model_load(path='./static/model', custom_objects={'InstanceNormalization': InstanceNormalization, 'ReflectionPadding2D':ReflectionPadding2D()}):
	model = load_model('./static/model/generatorB2A.h5', custom_objects=custom_objects)
	return model

def read_image(image_path):
	'Read an image from a file as an array'
	image = plt.imread(image_path, 'RGB').astype(np.float)
	return image

def load_image(image_path):
	image_shape = (128,128)
	images = []
	'Read and prepare image'
	image = read_image(image_path)
	image = cv2.resize(image, image_shape)
	images.append(image)
	images = np.array(images)/127.5 - 1.
	return images
	#print(images.shape)

def predict(image_path):
	model = model_load()
	image = load_image(image_path)
	prediction = model.predict(image)[0]
	prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
	PIL.Image.fromarray(prediction).save(image_path, 'jpeg')



