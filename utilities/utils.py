import os 
import zmq 
import argparse 

import pandas as pd 

import cv2 
import numpy as np 
import operator as op 
import itertools as it, functools as ft  

import pickle 

import tensorflow as tf 
import keras 

from os import path
from glob import glob 

from keras.applications.vgg16 import preprocess_input 
from imgaug import augmenters as iaa
from logger.log import logger 

def get_location(filename):
	return path.dirname(path.realpath(__file__))

def get_directories(target):
	return sorted(os.listdir(target))

def get_parser():
	return argparse.ArgumentParser()

def to_map(parser):
	return vars(parser.parse_args())

def pull_files(location, format='*'):
	return glob(path.join(location, format))

def load_vgg16(model_path):
	vgg16 = tf.keras.models.load_model(model_path, compile=False)
	vgg16_as_features_extractor = tf.keras.models.Model(
		inputs=vgg16.inputs, 
		outputs=vgg16.layers[-2].output 
	)
	return vgg16_as_features_extractor

def extract_features(image, extractor, size=(224, 224)):
	resized_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
	preprocessed_image = preprocess_input(resized_image)
	features = extractor.predict(preprocessed_image[None, :, :, :])
	return features  # shape : (1, 4096)

def build_classifier():
	return tf.keras.Sequential([
		tf.keras.Input(shape=(4096,)), 
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.Dense(units=128, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=1, activation='sigmoid'),
	])

def train_model(M, current_features, features_database):
	X = np.vstack(current_features + features_database)
	Y = np.hstack([np.ones(len(current_features)), np.zeros(len(features_database))])[:, None]

	M.compile(
		optimizer='adam',
		loss='binary_crossentropy',
		metrics=['accuracy']
	)

	M.fit(x=X, y=Y, epochs=64, batch_size=32, shuffle=True)

def get_similar_images(classifier, extractor, target_location):
	image_filenames = pull_files(target_location)
	scores = []
	for imf in image_filenames:
		print(imf)
		image = cv2.imread(imf, cv2.IMREAD_COLOR)
		image_features = extract_features(image, extractor)
		output = np.round(np.ravel(classifier.predict(image_features))[0], decimals=3)
		scores.append(output)

	location_score = [ list(tpl) for tpl in zip(image_filenames, scores) ]
	df = pd.DataFrame(columns=['path', 'score'], data=location_score)
	df['status'] = df['score'] >= 0.5  # high score of similarity due to DNN classifier
	df['name'] = df['path'].map(lambda elm: path.split(elm)[-1]) 
	return df 

def get_augmenter():
    seq = iaa.Sequential([
        iaa.OneOf([
        	iaa.Fliplr(0.5),
        	iaa.Flipud(0.5)
        ]), # horizontal flips | vertical flips
        iaa.Crop(percent=(0, 0.2)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.OneOf([
            	iaa.GaussianBlur(sigma=(0.5, 0.5)),
            	iaa.MedianBlur(k=(5, 5)), 
            	iaa.AverageBlur(k=(3, 3))
            ])
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.3),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.h
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.

        iaa.Sometimes(
        	0.5, 
        	iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        ),
        
        iaa.Sometimes(
        	0.5,
        	iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        ),
        iaa.Sometimes(
        	0.5,
        	iaa.Affine(rotate=(-30, 30))
        ),

		iaa.Sometimes(0.2, iaa.Grayscale(alpha=1.0))
    ], random_order=True) # apply augmenters in random order
    return seq 