import gzip
import struct
import numpy as np
import os

class MNISTLoader:
    def __init__(self, data_dir='mnist-images-augmented'):
        self.data_dir = data_dir

    def _load_images(self, image_path):
        # Try to open as gzip first, if that fails, open as regular file
        try:
            with gzip.open(image_path, 'rb') as f:
                magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        except (gzip.BadGzipFile, OSError):
            with open(image_path, 'rb') as f:
                magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        assert magic == 2051, f"Invalid magic number: {magic}"
        with open(image_path, 'rb') as f:
            f.read(16)  # Skip header
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
            return images.astype(np.float32) / 255.0

    def _load_labels(self, label_path):
        # Try to open as gzip first, if that fails, open as regular file
        try:
            with gzip.open(label_path, 'rb') as f:
                magic, num_labels = struct.unpack(">II", f.read(8))
        except (gzip.BadGzipFile, OSError):
            with open(label_path, 'rb') as f:
                magic, num_labels = struct.unpack(">II", f.read(8))
        
        assert magic == 2049, f"Invalid magic number: {magic}"
        with open(label_path, 'rb') as f:
            f.read(8)  # Skip header
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    def get_training_and_test_data(self):
        # Load training data
        train_images_path = os.path.join(self.data_dir, 'train-images-idx3-ubyte')
        train_labels_path = os.path.join(self.data_dir, 'train-labels-idx1-ubyte')
        X_train = self._load_images(train_images_path)
        Y_train = self._load_labels(train_labels_path)

        # Load test data
        #test_images_path = os.path.join(self.data_dir, 't10k-images-idx3-ubyte')
        #test_labels_path = os.path.join(self.data_dir, 't10k-labels-idx1-ubyte')
        #X_test = self._load_images(test_images_path)
        #Y_test = self._load_labels(test_labels_path)
        X_test = None
        Y_test = None

        return X_train, Y_train, X_test, Y_test 