import numpy as np
import random
from typing import List, Tuple, Optional, Callable, Iterator
import gc
import pandas as pd
import tensorflow as tf
from MNIST_data_handler.database_manager import Database_Manager

class MNISTDataset:
    """Dataset class for MNIST data with lazy loading capabilities"""
    
    def __init__(self, data_type: str = "train", transform: Optional[Callable] = None):
        """
        Initialize the MNIST dataset
        
        Args:
            data_type: Either "train" or "test"
            transform: Optional transform to apply to the data
        """
        self.data_type = data_type
        self.transform = transform
        self.data = None
        self.labels = None
        self.db = Database_Manager()
        self._user_data_loaded = False
        self._load_data()
    
    def _load_data(self):
        """Load the MNIST data from TensorFlow"""
        print(f"Loading {self.data_type} MNIST data using TensorFlow...")
        
        # Load MNIST data using TensorFlow
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        if self.data_type == "train":
            # Convert to float32 and normalize
            self.data = x_train.astype(np.float32) / 255.0
            self.labels = y_train.astype(np.int8)
            
            # Reshape to match the expected format (flatten to 784)
            self.data = self.data.reshape(-1, 784)
            
            # Clear original data
            del x_train, y_train
        else:  # test
            # Convert to float32 and normalize
            self.data = x_test.astype(np.float32) / 255.0
            self.labels = y_test.astype(np.int8)
            
            # Reshape to match the expected format (flatten to 784)
            self.data = self.data.reshape(-1, 784)
            
            # Clear original data
            del x_test, y_test
        
        gc.collect()
    
    def load_user_drawings(self):
        """Load user drawings and append to training data"""
        if self.data_type != "train" or self._user_data_loaded:
            return
            
        print("Loading user drawings...")
        userX_train, userY_train = self.db.get_user_drawings_data()
        
        if len(userX_train) > 0:
            # Convert to pandas DataFrame for efficient handling
            user_df = pd.DataFrame(userX_train, dtype=np.float32)
            user_labels = np.array(userY_train, dtype=np.int8)
            
            # Concatenate with existing data
            self.data = np.vstack((self.data, user_df.values))
            self.labels = np.concatenate((self.labels, user_labels))
            
            # Clear user data
            del userX_train, userY_train, user_df, user_labels
            gc.collect()
        
        self._user_data_loaded = True
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a single sample from the dataset"""
        sample = self.data[idx].copy()
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples from the dataset"""
        batch_data = self.data[indices].copy()
        batch_labels = self.labels[indices].copy()
        
        if self.transform:
            # Apply transform to each sample in the batch
            for i in range(len(batch_data)):
                batch_data[i] = self.transform(batch_data[i])
                
        return batch_data, batch_labels


class MNISTDataLoader:
    """DataLoader for MNIST data with batching and shuffling capabilities"""
    
    def __init__(self, dataset: MNISTDataset, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize the DataLoader
        
        Args:
            dataset: The dataset to load data from
            batch_size: The size of each batch
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Create an iterator that yields batches"""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_data, batch_labels = self.dataset.get_batch(batch_indices)
            yield batch_data, batch_labels
    
    def __len__(self) -> int:
        """Return the number of batches in the dataset"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# Augmentation functions
def random_shift_image(image: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """Apply random shift to an image"""
    import scipy.ndimage
    shift_x = np.random.randint(-max_shift, max_shift+1)
    shift_y = np.random.randint(-max_shift, max_shift+1)
    return scipy.ndimage.shift(image.reshape(28, 28), shift=(shift_x, shift_y), mode='constant', cval=0).flatten()

def random_rotate_image(image: np.ndarray, max_angle: float = 15) -> np.ndarray:
    """Apply random rotation to an image"""
    import scipy.ndimage
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy.ndimage.rotate(image.reshape(28, 28), angle, reshape=False, mode='constant', cval=0).flatten()

def add_gaussian_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add Gaussian noise to an image"""
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)


class AugmentationTransform:
    """Transform class for data augmentation"""
    
    def __init__(self, augment_factor: int = 1):
        """
        Initialize the augmentation transform
        
        Args:
            augment_factor: Number of augmented samples to generate per original sample
        """
        self.augment_factor = augment_factor
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to an image"""
        if random.random() < 0.5:
            image = random_shift_image(image)
        if random.random() < 0.5:
            image = random_rotate_image(image)
        if random.random() < 0.5:
            image = add_gaussian_noise(image)
        return image 