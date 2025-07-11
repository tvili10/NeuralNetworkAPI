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

        activations, zs = self.feed_forward_batch(X)
        delta = activations[-1] - Y  
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * leakyReLUDerivative(z)
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        
        nabla_w = [nw / batch_size for nw in nabla_w]
        nabla_b = [nb / batch_size for nb in nabla_b]
        return nabla_w, nabla_b

    def train(self, X_train, Y_train, epochs=5):
        num_of_batches = round(len(X_train) / 16)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            batchSize = len(X_train) // num_of_batches
            miniBatches = [(X_train[i * batchSize:(i + 1) * batchSize], 
                    Y_train[i * batchSize:(i + 1) * batchSize]) 
                   for i in range(num_of_batches)]
            
            for miniBatch in miniBatches:
                self.mini_batch_gradient_update(miniBatch, learningRate=0.01)

    def mini_batch_gradient_update(self, miniBatch, learningRate):
        X_batch = np.array(miniBatch[0]).T  
        Y_batch_indices = np.array(miniBatch[1], dtype=int)
        num_classes = self.biases[-1].shape[0]
        Y_batch = np.zeros((num_classes, X_batch.shape[1]))
        Y_batch[Y_batch_indices, np.arange(X_batch.shape[1])] = 1

        nabla_w, nabla_b = self.backpropagation_batch(X_batch, Y_batch)
        self.weights = [w - learningRate * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learningRate * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activations, zs = self.feed_forward(x)
        outputLayer = np.zeros((activations[-1].shape[0], 1))
        outputLayer[y] = 1

        delta =  (activations[-1] - outputLayer)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.weights) + 1):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].T, delta) * leakyReLUDerivative(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (nabla_w, nabla_b)

    def test(self, X_test, Y_test):
        correct = 0
        for x, y in zip(X_test, Y_test):
            activations, _ = self.feed_forward(x)
            if np.argmax(activations[-1]) == y:
                correct += 1
        print(f"Accuracy: {correct / len(X_test) * 100}%")


def leakyReLU(x):
    return np.maximum(0.1 * x, x)

def leakyReLUDerivative(x):
    return np.where(x > 0, 1, 0.1)