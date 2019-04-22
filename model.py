import sklearn, h5py
import numpy as np
import tensorflow as tf

class Model_SVM:
    
    def __init__(self, labels, features, test_size):
        self.labels = labels
        self.features = features
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.test_size = test_size
        self.model = sklearn.svm.SVC(kernel="linear", gamma="scale", C = 10)

    def split_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.features, self.labels, test_size=self.test_size, random_state=np.random)

    def train(self):
        self.model.fit(self.x_train, self.y_train)
    
    def get_score(self):
        return self.model.score(self.x_test, self.y_test)

class cnn:
    
    def __init__(self, labels, features, test_size):
        self.labels = labels
        self.features = features
        self.test_size = test_size
    

