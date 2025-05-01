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

    def get_training_and_test_data(self, augment=False):
        X_train, Y_train, X_test, Y_test = self.load_mnist_data()
        
        if augment:
            # Process augmented data in batches and concatenate
            augmented_batches = []
            label_batches = []
            for aug_images, aug_labels in self.augment_mnist_images(X_train, Y_train):
                augmented_batches.append(aug_images)
                label_batches.append(aug_labels)
            
            X_train = np.concatenate(augmented_batches, axis=0)
            Y_train = np.concatenate(label_batches, axis=0)
            
        return X_train, Y_train, X_test, Y_test
    
    def get_training_data(self):
        X_train, Y_train, _, _ = self.load_mnist_data()
        return X_train, Y_train

    def add_data(self, pixels, label):
        self.db.add_data(pixels, label)

    def augment_mnist_images(self, images, labels, augment_factor=2, batch_size=1000):
        
        num_images = len(images)
        for i in range(0, num_images, batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            augmented_batch_images = []
            augmented_batch_labels = []
            
            for img, lbl in zip(batch_images, batch_labels):
                # Add original image
                augmented_batch_images.append(img)
                augmented_batch_labels.append(lbl)
                
                # Generate augmented versions
                for _ in range(augment_factor):
                    aug_img = img.copy()
                    
                    if random.random() < 0.5:
                        aug_img = self.random_shift_image(aug_img)
                    if random.random() < 0.5:
                        aug_img = self.random_rotate_image(aug_img)
                    if random.random() < 0.5:
                        aug_img = self.add_gaussian_noise(aug_img)
                    
                    augmented_batch_images.append(aug_img)
                    augmented_batch_labels.append(lbl)
            
            yield np.array(augmented_batch_images), np.array(augmented_batch_labels)

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

####################################