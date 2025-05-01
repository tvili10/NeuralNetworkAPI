import numpy as np
import random
from MNIST_data_handler.database_manager import Database_Manager
import scipy.ndimage
from MNIST_data_handler.mnist_loader import MNISTLoader

class Datahandler:
    def __init__(self):        
        self.db = Database_Manager()
        self.mnist_loader = MNISTLoader()
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._augmented = False

    def load_mnist_data(self):
        if self._X_train is None:
            self._X_train, self._Y_train, self._X_test, self._Y_test = self.mnist_loader.load_mnist_data()
            
            # Load user drawings
            userX_train, userY_train = self.db.get_user_drawings_data()
            if len(userX_train) > 0:
                self._X_train = np.concatenate((self._X_train, userX_train), axis=0)
                self._Y_train = np.concatenate((self._Y_train, userY_train), axis=0)
        
        return self._X_train, self._Y_train, self._X_test, self._Y_test

    def get_training_and_test_data(self, augment=False):
        X_train, Y_train, X_test, Y_test = self.load_mnist_data()
        
        if augment and not self._augmented:
            X_train, Y_train = augment_mnist_images(X_train, Y_train)
            self._augmented = True
            self._X_train = X_train
            self._Y_train = Y_train
            
        return X_train, Y_train, X_test, Y_test
    
    def add_data(self, pixels, label):
        self.db.add_data(pixels, label)
        # Reset cached data to force reload with new example
        self._X_train = None
        self._Y_train = None
        self._augmented = False


### image manipulation functions ###
def random_shift_image(image, max_shift=3):
    shift_x = np.random.randint(-max_shift, max_shift+1)
    shift_y = np.random.randint(-max_shift, max_shift+1)
    return scipy.ndimage.shift(image.reshape(28, 28), shift=(shift_x, shift_y), mode='constant', cval=0).flatten()

def random_rotate_image(image, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy.ndimage.rotate(image.reshape(28, 28), angle, reshape=False, mode='constant', cval=0).flatten()

def add_gaussian_noise(image, noise_level=0.1):
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)  

def augment_mnist_images(images, labels, augment_factor=2):
    augmented_images = []
    augmented_labels = []
    
    for img, lbl in zip(images, labels):
        augmented_images.append(img)  
        augmented_labels.append(lbl)
        
        for _ in range(augment_factor):
            aug_img = img.copy()
            
            if random.random() < 0.5:
                aug_img = random_shift_image(aug_img)
            if random.random() < 0.5:
                aug_img = random_rotate_image(aug_img)
            if random.random() < 0.5:
                aug_img = add_gaussian_noise(aug_img)
            
            augmented_images.append(aug_img)
            augmented_labels.append(lbl)
    
    return np.array(augmented_images), np.array(augmented_labels)

####################################