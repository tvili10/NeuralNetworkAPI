import numpy as np
import random
import scipy.ndimage
from typing import Tuple, List
import gc
import pandas as pd
import tensorflow as tf
from MNIST_data_handler.database_manager import Database_Manager
from MNIST_data_handler.dataloader import MNISTDataset, MNISTDataLoader, AugmentationTransform

class Datahandler:
    def __init__(self):        
        self.db = Database_Manager()
        self._train_dataset = None
        self._test_dataset = None
        self._augmented = False
        self._user_data_loaded = False

    def load_mnist_data(self) -> Tuple[MNISTDataset, MNISTDataset]:
        """Load MNIST data using the dataset classes"""
        if self._train_dataset is None:
            print("Loading MNIST data using TensorFlow...")
            
            # Create datasets
            self._train_dataset = MNISTDataset(data_type="train")
            self._test_dataset = MNISTDataset(data_type="test")
            
            # Load user drawings if needed
            if not self._user_data_loaded:
                self._load_user_drawings()
        
        return self._train_dataset, self._test_dataset

    def _load_user_drawings(self):
        """Load user drawings and append to training data"""
        if self._train_dataset is not None:
            self._train_dataset.load_user_drawings()
            self._user_data_loaded = True

    def get_training_and_test_data(self, augment=False, batch_size=32) -> Tuple[MNISTDataLoader, MNISTDataLoader]:
        """Get training and test dataloaders"""
        train_dataset, test_dataset = self.load_mnist_data()
        
        # Apply augmentation if requested
        if augment and not self._augmented:
            print("Creating augmented training dataset...")
            train_dataset.transform = AugmentationTransform()
            self._augmented = True
            
        # Create dataloaders
        train_loader = MNISTDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = MNISTDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
        return train_loader, test_loader
    
    def add_data(self, pixels: List[float], label: int):
        """Add a new training example to the database"""
        self.db.add_data(pixels, label)
        # Reset cached data to force reload with new example
        self._train_dataset = None
        self._test_dataset = None
        self._augmented = False
        self._user_data_loaded = False
        gc.collect()

# Keep these functions for backward compatibility
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