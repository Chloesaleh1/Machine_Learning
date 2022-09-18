from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNN(object):
    def __init__(self):
        # change these to appropriate values

        self.batch_size = 90
        self.epochs = 12
        self.init_lr= 0.1 #learning rate

        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model.
        Then, use model.add() to build layers in your own model
        Return: model
        '''

        #TODO: implement this

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(16, (3,3), padding='same', input_shape=(32,32,3)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))

        model.add(tf.keras.layers.Dropout(0.25, noise_shape=None, seed=None))
        model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
        model.add(tf.keras.layers.Dropout(0.25, noise_shape=None, seed=None))
        model.add(Flatten())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))
        model.add(tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None))
        model.add(tf.keras.layers.Dense(10))
        model.add(tf.keras.layers.Activation('softmax'))
        self.model=model

        return self.model


    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model

        #TODO: implement this
        self.model = model
        self.model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model
