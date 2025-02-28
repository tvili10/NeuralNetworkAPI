import numpy as np
import random
from mnist.loader import MNIST
from MNIST_data_handler.database_manager import Database_Manager
import scipy.ndimage

from MNIST_data_handler.database_manager import Database_Manager

class Datahandler:
    def __init__(self):        
        self.db = Database_Manager()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_mnist_data()

        self.X_train, self.Y_train = augment_mnist_images(self.X_train, self.Y_train)

        userX_train, userY_train = self.db.get_user_drawings_data()

        ### Combine training data ###
        self.X_train = np.concatenate((self.X_train, userX_train), axis=0)
        self.Y_train = np.concatenate((self.Y_train, userY_train), axis=0)
        

    def load_mnist_data(self):
        mndata = MNIST('data')
        trainingImages, trainingLabels = mndata.load_training()
        trainingImages = np.array(trainingImages).astype(np.float32) / 255.0
        testImages, testLabels = mndata.load_testing()
        testImages = np.array(testImages).astype(np.float32) / 255.0
        return trainingImages, trainingLabels, testImages, testLabels
    

    def get_training_and_test_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test
    
    def add_data(self, pixels, label):
        self.db.add_data(pixels, label)


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