import numpy as np
import random
import scipy.ndimage
from typing import Tuple, List
import gc
import pandas as pd
import tensorflow as tf
from MNIST_data_handler.database_manager import Database_Manager

class Datahandler:
    def __init__(self):        
        self.db = Database_Manager()
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._augmented = False
        self._user_data_loaded = False

    def load_mnist_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._X_train is None:
            print("Loading MNIST data using TensorFlow...")
            
            # Load MNIST data using TensorFlow
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            # Convert to float32 and normalize
            self._X_train = x_train.astype(np.float32) / 255.0
            self._Y_train = y_train.astype(np.int8)
            self._X_test = x_test.astype(np.float32) / 255.0
            self._Y_test = y_test.astype(np.int8)
            
            # Reshape to match the expected format (flatten to 784)
            self._X_train = self._X_train.reshape(-1, 784)
            self._X_test = self._X_test.reshape(-1, 784)
            
            # Clear original data
            del x_train, y_train, x_test, y_test
            gc.collect()
            
            # Load user drawings only if needed
            if not self._user_data_loaded:
                self._load_user_drawings()
        
        return self._X_train, self._Y_train, self._X_test, self._Y_test

    def _load_user_drawings(self):
        """Load user drawings and append to training data"""
        print("Loading user drawings...")
        userX_train, userY_train = self.db.get_user_drawings_data()
        
        if len(userX_train) > 0:
            # Convert to pandas DataFrame for efficient handling
            user_df = pd.DataFrame(userX_train, dtype=np.float32)
            user_labels = np.array(userY_train, dtype=np.int8)
            
            # Concatenate with existing data
            self._X_train = np.vstack((self._X_train, user_df.values))
            self._Y_train = np.concatenate((self._Y_train, user_labels))
            
            # Clear user data
            del userX_train, userY_train, user_df, user_labels
            gc.collect()
        
        self._user_data_loaded = True

    def get_training_and_test_data(self, augment=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, Y_train, X_test, Y_test = self.load_mnist_data()
        
        if augment and not self._augmented:
            print("Augmenting training data...")
            X_train, Y_train = augment_mnist_images(X_train, Y_train)
            self._augmented = True
            self._X_train = X_train
            self._Y_train = Y_train
            
        return X_train, Y_train, X_test, Y_test
    
    def add_data(self, pixels: List[float], label: int):
        self.db.add_data(pixels, label)
        # Reset cached data to force reload with new example
        self._X_train = None
        self._Y_train = None
        self._augmented = False
        self._user_data_loaded = False
        gc.collect()

def random_shift_image(image: np.ndarray, max_shift: int = 3) -> np.ndarray:
    shift_x = np.random.randint(-max_shift, max_shift+1)
    shift_y = np.random.randint(-max_shift, max_shift+1)
    return scipy.ndimage.shift(image.reshape(28, 28), shift=(shift_x, shift_y), mode='constant', cval=0).flatten()

def random_rotate_image(image: np.ndarray, max_angle: float = 15) -> np.ndarray:
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy.ndimage.rotate(image.reshape(28, 28), angle, reshape=False, mode='constant', cval=0).flatten()

def add_gaussian_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)  

def augment_mnist_images(images: np.ndarray, labels: np.ndarray, augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    print("Starting augmentation...")
    total_images = len(images) * (augment_factor + 1)
    
    # Use pandas for efficient memory management during augmentation
    augmented_df = pd.DataFrame(np.zeros((total_images, 784)), dtype=np.float32)
    augmented_labels = np.zeros(total_images, dtype=np.int8)
    
    # Copy original images
    augmented_df.iloc[:len(images)] = images
    augmented_labels[:len(images)] = labels
    
    current_idx = len(images)
    
    # Generate augmented images
    for i, (img, lbl) in enumerate(zip(images, labels)):
        for _ in range(augment_factor):
            aug_img = img.copy()
            
            if random.random() < 0.5:
                aug_img = random_shift_image(aug_img)
            if random.random() < 0.5:
                aug_img = random_rotate_image(aug_img)
            if random.random() < 0.5:
                aug_img = add_gaussian_noise(aug_img)
            
            augmented_df.iloc[current_idx] = aug_img
            augmented_labels[current_idx] = lbl
            current_idx += 1
            
        if i % 1000 == 0:
            print(f"Augmented {i}/{len(images)} images")
            gc.collect()  # Periodic garbage collection during long operation
    
    print("Augmentation complete!")
    return augmented_df.values, augmented_labels

####################################