import numpy as np
import features as ftr

class FeatureNetwork():
    def __init__(self, sizes, features):
        """Features is a list of numpy arrays containing template
            image for each feature
            Each template image is a tuple of feature and antifeature"""
        self.sizes = sizes
        self.nlayers = len(sizes)
        self.features = [ftr.Feature(feat, antifeat) for feat, antifeat in features]
        self.weights = [np.random.randn(si, ze) for si, ze in zip(sizes[:-1], sizes[1:])]
        self.biases = np.random.randn(1, size) for size in sizes[1:]
        self.weights[0] = np.concatenate([f.weights for f in self.features])
        self.biases[0] = np.concatenate([f.biases for bias in self.features])

class CrossEntropyNetwork():

    def __init__(self, sizes):
        """Sizes is a list of the number of neurons in each layer.
            nlayers is the number of layers.
            biases is a list of bias vectors for every non-input layer
            weights is a list of weight matrices for every non-input layer
            zs is a list of of weighted input vectors to every non-input layer
            avs is a list of activations for every layer"""
        self.sizes = sizes
        self.nlayers = len(sizes)
        self.biases = [np.random.randn(1,size) for size in sizes[1:]]
        self.weights = [np.random.randn(si,ze) for si, ze in zip(sizes[:-1],sizes[1:])]
        self.zs = [np.zeros([1,size]) for size in sizes[1:]]
        self.avs = [np.zeros([1,size]) for size in sizes]

    def feedforward(self, a):
        """Updates zs and avs with given input and current weights/biases"""
        self.avs[0] = a.reshape([1, a.shape[0]])
        for i in range(self.nlayers)[:-1]:
            self.zs[i] = np.dot(self.avs[i], self.weights[i]) + self.biases[i]
            self.avs[i+1] = sigmoidv(self.zs[i])
        return self.avs[-1]

    def cost_derivative(self, A, Y):
        """Derivative of quadratic cost function"""
        return A - Y

    def SGD(self, train, eta, epochs, mbatch_size, test=None):
        """Neural network learns by stochastic gradient descent algorithm.
            train is training data as a list of tuples (input, value)
            eta is the learning rate
            epochs is the number of times to run SGD
            mbatch_size is the size of each minibatch
            test is test data as a list of tuples (input, value) (optional)"""
        if test:
            ntest = len(test)
        ntrain = len(train)
        for epoch in range(epochs):
            np.random.shuffle(train)
            mbatches = [train[k:k+mbatch_size] for k in range(0, ntrain, mbatch_size)]
            for mbatch in mbatches:
                self.update_mbatch(mbatch, eta)
#             if test:
#                 print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test), ntest))
#             else:
#                 print("Epoch {0} complete".format(epoch))
        print("SGD complete")

    def update_mbatch(self, mbatch, eta):
        """Update the weights and biases in a minibatch"""
        nablaB = [np.zeros_like(b) for b in self.biases] # dC/db by layer
        nablaW = [np.zeros_like(w) for w in self.weights] # dC/dw by layer
        for x, y in mbatch:
            dnablaB, dnablaW = self.backprop(x,y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, dnablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, dnablaW)]
        self.weights = [w - (eta/len(mbatch))*nw for w, nw in zip(self.weights, nablaW)]
        # take small steps in opposite direction from gradient of cost function
        self.biases = [b - (eta/len(mbatch))*nb for b, nb in zip(self.biases, nablaB)]

    def backprop(self, x, y):
        """Calculate the gradients after a small perturbation to activations"""
        nablaB = [np.zeros_like(b) for b in self.biases]
        nablaW = [np.zeros_like(w) for w in self.weights]
        self.avs[0] = x.reshape([1,x.size])
        for i in range(self.nlayers)[:-1]:
            self.zs[i] = np.dot(self.avs[i], self.weights[i]) + self.biases[i]
            self.avs[i+1] = sigmoidv(self.zs[i])
            self.avs[i+1] = self.avs[i+1].reshape([1,self.avs[i+1].size])
        delta = self.cost_derivative(self.avs[-1], y) * dsigdzv(self.zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(self.avs[-2].T, delta)
        for layer in range(2, self.nlayers):
            spv = dsigdz(self.zs[-layer])
            delta = np.dot(delta, self.weights[-layer+1].T) * spv
            nablaB[-layer] = delta
            nablaW[-layer] = np.dot(self.avs[-layer-1].T, delta)
        return(nablaB, nablaW)

    def evaluate(self, test):
        """Count number of test items for which network gets it right (eg, gives maximum value)"""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test]
        return sum(int(x == y) for (x, y) in test_results)







class QuadNetwork():
    """Implements a very crude feedforward neural network.
        Neurons are sigmoid.
        SGD is used for learning with quadratic cost."""

    def __init__(self, sizes):
        """Sizes is a list of the number of neurons in each layer.
            nlayers is the number of layers.
            biases is a list of bias vectors for every non-input layer
            weights is a list of weight matrices for every non-input layer
            zs is a list of of weighted input vectors to every non-input layer
            avs is a list of activations for every layer"""
        self.sizes = sizes
        self.nlayers = len(sizes)
        self.biases = [np.random.randn(1,size) for size in sizes[1:]]
        self.weights = [np.random.randn(si,ze) for si, ze in zip(sizes[:-1],sizes[1:])]
        self.zs = [np.zeros([1,size]) for size in sizes[1:]]
        self.avs = [np.zeros([1,size]) for size in sizes]

    def feedforward(self, a):
        """Updates zs and avs with given input and current weights/biases"""
        self.avs[0] = a.reshape([1, a.shape[0]])
        for i in range(self.nlayers)[:-1]:
            self.zs[i] = np.dot(self.avs[i], self.weights[i]) + self.biases[i]
            self.avs[i+1] = sigmoidv(self.zs[i])
        return self.avs[-1]

    def cost_derivative(self, A, Y):
        """Derivative of quadratic cost function"""
        return A - Y

    def SGD(self, train, eta, epochs, mbatch_size, test=None):
        """Neural network learns by stochastic gradient descent algorithm.
            train is training data as a list of tuples (input, value)
            eta is the learning rate
            epochs is the number of times to run SGD
            mbatch_size is the size of each minibatch
            test is test data as a list of tuples (input, value) (optional)"""
        if test:
            ntest = len(test)
        ntrain = len(train)
        for epoch in range(epochs):
            np.random.shuffle(train)
            mbatches = [train[k:k+mbatch_size] for k in range(0, ntrain, mbatch_size)]
            for mbatch in mbatches:
                self.update_mbatch(mbatch, eta)
#             if test:
#                 print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test), ntest))
#             else:
#                 print("Epoch {0} complete".format(epoch))
        print("SGD complete")

    def update_mbatch(self, mbatch, eta):
        """Update the weights and biases in a minibatch"""
        nablaB = [np.zeros_like(b) for b in self.biases] # dC/db by layer
        nablaW = [np.zeros_like(w) for w in self.weights] # dC/dw by layer
        for x, y in mbatch:
            dnablaB, dnablaW = self.backprop(x,y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, dnablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, dnablaW)]
        self.weights = [w - (eta/len(mbatch))*nw for w, nw in zip(self.weights, nablaW)]
        # take small steps in opposite direction from gradient of cost function
        self.biases = [b - (eta/len(mbatch))*nb for b, nb in zip(self.biases, nablaB)]

    def backprop(self, x, y):
        """Calculate the gradients after a small perturbation to activations"""
        nablaB = [np.zeros_like(b) for b in self.biases]
        nablaW = [np.zeros_like(w) for w in self.weights]
        self.avs[0] = x.reshape([1,x.size])
        for i in range(self.nlayers)[:-1]:
            self.zs[i] = np.dot(self.avs[i], self.weights[i]) + self.biases[i]
            self.avs[i+1] = sigmoidv(self.zs[i])
            self.avs[i+1] = self.avs[i+1].reshape([1,self.avs[i+1].size])
        delta = self.cost_derivative(self.avs[-1], y) * dsigdzv(self.zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(self.avs[-2].T, delta)
        for layer in range(2, self.nlayers):
            spv = dsigdz(self.zs[-layer])
            delta = np.dot(delta, self.weights[-layer+1].T) * spv
            nablaB[-layer] = delta
            nablaW[-layer] = np.dot(self.avs[-layer-1].T, delta)
        return(nablaB, nablaW)

    def evaluate(self, test):
        """Count number of test items for which network gets it right (eg, gives maximum value)"""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test]
        return sum(int(x == y) for (x, y) in test_results)

# helper funcs
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def dsigdz(z):
    return sigmoid(z) * (1-sigmoid(z))
sigmoidv = np.vectorize(sigmoid)
dsigdzv = np.vectorize(dsigdz)
