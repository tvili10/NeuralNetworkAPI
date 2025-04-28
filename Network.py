import numpy as np
from typing import List, Tuple, Iterator


class MultilayerPerceptron:
    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) * np.sqrt(1.0 / x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    
    def feed_forward(self, A):
        A = np.array(A).reshape(-1, 1)
        activations = [A]
        zValues = []
        for W, B in zip(self.weights, self.biases):
            z = np.dot(W, A) + B
            zValues.append(z)
            A = leakyReLU(z)
            activations.append(A)
        activations[-1] = np.exp(activations[-1]) / np.sum(np.exp(activations[-1]))
        return activations, zValues

    def feed_forward_batch(self, X):
        activations = [X]
        zValues = []
        A = X
        for W, B in zip(self.weights, self.biases):
            z = np.dot(W, A) + B 
            zValues.append(z)
            A = leakyReLU(z)
            activations.append(A)
        
        # Apply softmax to the output layer
        activations[-1] = np.exp(activations[-1]) / np.sum(np.exp(activations[-1]), axis=0, keepdims=True)
        return activations, zValues

    def backpropagation_batch(self, X, Y):
        batch_size = X.shape[1]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Forward pass
        activations, zs = self.feed_forward_batch(X)
        
        # Backward pass
        delta = activations[-1] - Y  
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * leakyReLUDerivative(z)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        
        # Clear intermediate results
        del activations, zs, delta
        
        nabla_w = [nw / batch_size for nw in nabla_w]
        nabla_b = [nb / batch_size for nb in nabla_b]
        return nabla_w, nabla_b

    def train(self, train_loader, epochs=5, learning_rate=0.01):
        """
        Train the network using a dataloader
        
        Args:
            train_loader: DataLoader providing training batches
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Process mini-batches from the dataloader
            for batch_idx, (X_batch, Y_batch_indices) in enumerate(train_loader):
                # Transpose X_batch to match the expected format
                X_batch = X_batch.T
                
                # Create one-hot encoded labels
                num_classes = self.biases[-1].shape[0]
                Y_batch = np.zeros((num_classes, X_batch.shape[1]))
                Y_batch[Y_batch_indices, np.arange(X_batch.shape[1])] = 1
                
                # Update weights and biases
                nabla_w, nabla_b = self.backpropagation_batch(X_batch, Y_batch)
                self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_b)]
                
                # Clear batch data
                del X_batch, Y_batch, Y_batch_indices, nabla_w, nabla_b
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}")

    def test(self, test_loader):
        """
        Test the network using a dataloader
        
        Args:
            test_loader: DataLoader providing test batches
        """
        correct = 0
        total = 0
        
        for X_batch, Y_batch in test_loader:
            # Transpose X_batch to match the expected format
            X_batch = X_batch.T
            
            # Forward pass
            activations, _ = self.feed_forward_batch(X_batch)
            predictions = np.argmax(activations[-1], axis=0)
            
            # Calculate accuracy
            correct += np.sum(predictions == Y_batch)
            total += len(Y_batch)
            
            # Clear batch data
            del X_batch, activations, predictions
        
        accuracy = correct / total * 100
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy


def leakyReLU(x):
    return np.maximum(0.1 * x, x)

def leakyReLUDerivative(x):
    return np.where(x > 0, 1, 0.1)