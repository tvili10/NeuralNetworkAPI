import numpy as np
import random
from mnist.loader import MNIST
from database_manager import Database_Manager
import scipy.ndimage

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

class NumberReognizer:
    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) * np.sqrt(1.0 / x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.trainingImages, self.trainingLabels, self.testImages, self.testLabels = self.loadMNSITData()
        
        self.trainingImages, self.trainingLabels = augment_mnist_images(self.trainingImages, self.trainingLabels)

        self.db = Database_Manager()

    
    def loadMNSITData(self):
        mndata = MNIST('data')
        trainingImages, trainingLabels = mndata.load_training()
        trainingImages = np.array(trainingImages).astype(np.float32) / 255.0
        testImages, testLabels = mndata.load_testing()
        testImages = np.array(testImages).astype(np.float32) / 255.0
        return trainingImages, trainingLabels, testImages, testLabels
    
    def addTrainingData(self, pixels, label):
        self.db.add_data(pixels, label)

    def feedForward(self, A):
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

    def trainOnMNISTDataSet(self, epochs=20):
        data = list(zip(self.trainingImages, self.trainingLabels))
        random.shuffle(data)
        self.trainingImages, self.trainingLabels = zip(*data)

        numberOfBatches = round(len(self.trainingImages) / 32)

        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            batchSize = len(self.trainingImages) // numberOfBatches
            miniBatches = [(self.trainingImages[i * batchSize:(i + 1) * batchSize], 
                        self.trainingLabels[i * batchSize:(i + 1) * batchSize]) 
                       for i in range(numberOfBatches)]
            for miniBatch in miniBatches:
                self.miniBatchGradientUpdate(miniBatch, learningRate=0.01)
            
        self.testNetwork() 

    def miniBatchGradientUpdate(self, miniBatch, learningRate):

    
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
    
    
        for x, y in zip(miniBatch[0], miniBatch[1]):
            delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        batch_size = len(miniBatch[0])   
        self.weights = [w - (learningRate * (nw / batch_size)) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learningRate * (nb / batch_size)) for b, nb in zip(self.biases, nabla_b)]

        

    def backpropagation(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activations, zs = self.feedForward(x)
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

    def testNetwork(self):
        
        correct = 0
        for x, y in zip(self.testImages, self.testLabels):
            activations, _ = self.feedForward(x)
            if np.argmax(activations[-1]) == y:
                correct += 1
        print(f"Accuracy: {correct / len(self.testImages) * 100}%")


def leakyReLU(x):
    return np.maximum(0.1 * x, x)

def leakyReLUDerivative(x):
    return np.where(x > 0, 1, 0.1)