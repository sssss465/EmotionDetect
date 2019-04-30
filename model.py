import sklearn
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from time import time
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint


class Model_SVM:

    def __init__(self, labels, features, test_size):
        self.labels = labels
        self.features = features
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.test_size = test_size
        self.model = sklearn.svm.SVC(kernel="linear", gamma="scale", C=10)

    def split_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=np.random)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def get_score(self):
        return self.model.score(self.x_test, self.y_test)


class cnn:
    def __init__(self, labels, features, test_size, num_classes=8, epochs=500,
                 batch_size=64):  # not sure if ideal // hyperparmeter tuning needed
        self.labels = labels
        self.features = features
        self.test_size = test_size
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

    def split_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=np.random)

    def build_model(self):
        layers = keras.layers
        # we need to transform x and turn y into one hot
        self.x_train = self.x_train[..., np.newaxis] # add new dimension
        self.x_test = self.x_test[..., np.newaxis] 
        # self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        # self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes) 
        self.model = keras.Sequential([
            layers.Conv1d(32, (3, 3), padding='same',
                          activation=tf.nn.relu, input_shape=self.x_train.shape[1:]),
            layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # layers.BatchNormalization(),

            layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
            layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])  # model based off cifar10 dataset, should be deep enough
        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        print("built cnn model..")

    def train(self):
        print("training model")
        # may want to change this
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'keras_cnn_trained_model.h5'
        # model_name = 'cnn_trained_model-{epoch:02d}-{val_acc:.2f}.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        if os.path.isfile(model_path):
            self.model.load_weights(model_path)
        # may want to change save_best to true
        checkpoint = ModelCheckpoint(
            model_path, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True,
                                  write_images=True, histogram_freq=0)
        self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(self.x_test, self.y_test),
                  shuffle=True,
                  callbacks=[tensorboard, checkpoint])

    def evaluate(self):
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
