import numpy as np
import random
from MNIST_data_handler.database_manager import Database_Manager
import scipy.ndimage
from MNIST_data_handler.mnist_loader import MNISTLoader

class Datahandler:
    def __init__(self):        
        self.db = Database_Manager()
        self.mnist_loader = MNISTLoader()

    def load_mnist_data(self):
        X_train, Y_train, X_test, Y_test = self.mnist_loader.get_training_and_test_data()
        
        # Load user drawings
        userX_train, userY_train = self.db.get_user_drawings_data()
        if len(userX_train) > 0:
            X_train = np.concatenate((X_train, userX_train), axis=0)
            Y_train = np.concatenate((Y_train, userY_train), axis=0)
        
        return X_train, Y_train, X_test, Y_test

    def get_training_and_test_data(self):
        X_train, Y_train, X_test, Y_test = self.load_mnist_data()
        
          
        return X_train, Y_train, X_test, Y_test
    
    def get_training_data(self):
        X_train, Y_train, _, _ = self.load_mnist_data()
        return X_train, Y_train

    def add_data(self, pixels, label):
        self.db.add_data(pixels, label)

