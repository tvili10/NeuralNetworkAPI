import numpy as np


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

    def train(self, X_train, Y_train, epochs=5):
        num_of_batches = round(len(X_train) / 16)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            batchSize = len(X_train) // num_of_batches

            
            # Process mini-batches one at a time
            for i in range(num_of_batches):
                start_idx = i * batchSize
                end_idx = (i + 1) * batchSize
                
                X_batch = np.array(X_train[start_idx:end_idx]).T
                Y_batch_indices = np.array(Y_train[start_idx:end_idx], dtype=int)
                
                # Create one-hot encoded labels more efficiently
                num_classes = self.biases[-1].shape[0]
                Y_batch = np.zeros((num_classes, X_batch.shape[1]))
                Y_batch[Y_batch_indices, np.arange(X_batch.shape[1])] = 1
                
                # Update weights and biases
                nabla_w, nabla_b = self.backpropagation_batch(X_batch, Y_batch)
                self.weights = [w - 0.01 * nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b - 0.01 * nb for b, nb in zip(self.biases, nabla_b)]
                
                # Clear batch data
                del X_batch, Y_batch, Y_batch_indices, nabla_w, nabla_b

    def test(self, X_test, Y_test):
        correct = 0
        total = len(X_test)
        batch_size = 100  # Process test data in smaller batches
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            X_batch = np.array(X_test[i:batch_end]).T
            Y_batch = Y_test[i:batch_end]
            
            activations, _ = self.feed_forward_batch(X_batch)
            predictions = np.argmax(activations[-1], axis=0)
            correct += np.sum(predictions == Y_batch)
            
            # Clear batch data
            del X_batch, activations, predictions
        
        print(f"Accuracy: {correct / total * 100}%")


def leakyReLU(x):
    return np.maximum(0.1 * x, x)

def leakyReLUDerivative(x):
    return np.where(x > 0, 1, 0.1)