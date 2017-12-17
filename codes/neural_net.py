import numpy as np
import mnist_loader
import sys
import numpy
from sklearn.preprocessing import OneHotEncoder
numpy.set_printoptions(threshold=sys.maxsize)

class Neural_Net(object):

    #def __init__(self, X_train, Y_train, nn_dims, init_method='init_with_rands'):
    def __init__(self, nn_dims, init_method='init_with_rands'):
        #self.X = X_train
        #self.Y = Y_train
        self.L = len(nn_dims)
        #self.nx = X_train.shape[0]
        #self.nh = nh
        #self.ny = Y_train.shape[0]
        #temporary
        self.params = {}
        self.activations = {}
        self.Z = {}
        self.init_with_rands(nn_dims)

    def init_with_zeroBias(self):
        self.w1 = np.random.randn(self.nh,self.nx) * 0.01
        self.b1 = np.zeros((self.nh, 1))
        self.w2 = np.random.randn(self.ny,self.nh) * 0.01
        self.b2 = np.zeros((self.ny,1))

    def init_with_rands(self, nn_dims):
        for l in range(1,self.L):
            self.params['W'+str(l)] = np.random.randn(nn_dims[l],nn_dims[l-1]) * 0.01
            self.params['b'+str(l)] = np.random.randn(nn_dims[l],1) * 0.01

    def init_better(self):
        self.w1 = np.zeros((nh,nx))

    def forward_propagation(self, X):
        #for b,w in zip(self.Bs, self.Ws):
        self.activations[0] = X
        L = self.L - 1
        for l in range(1, L):
            self.Z[l] = np.dot(self.params['W'+str(l)], self.activations[l-1]) + self.params['b'+str(l)]
            self.activations[l] = self.relu(self.Z[l])

        temp = np.exp(np.dot(self.params['W'+str(L)], self.activations[L-1]) + self.params['b'+str(L)])
        self.activations[L] = temp / np.sum(temp, axis=0)
        #self.activations[L-1] = self.sigmoid(np.dot(self.params['w'+str(L-1)], self.activations[L-2]) + self.params['b'+str(L-1)])
            #a1 = np.tanh(np.dot(self.w1, X_test) + self.b1)
            #a2 = self.sigmoid(np.dot(self.w2, self.a1) + self.b2)

    def backward_propagatation(self, Y_nn, Y):
        L = self.L - 1
        grads = {}
        m = Y.shape[1]
        dAL = -np.divide(Y, Y_nn)
        dZL = Y_nn-Y
        grads['dW'+str(L)] = np.dot(dZL, self.activations[L-1].T) / m
        grads['db'+str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.params['W'+str(L)].T, dZL)
        for l in reversed(range(1,L)):
            dA = dA_prev
            dZ = np.array(dA, copy=True)
            dZ[self.Z[l]<=0] = 0
            grads['dW'+str(l)] = np.dot(dZ, self.activations[l-1].T) / m
            grads['db'+str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(self.params['W'+str(l)].T, dZ)

        return grads


    def nn_training(self, X_train, Y_train, iterations, alpha):
        for i in range(0, iterations):
            self.forward_propagation(X_train)
            cost = self.evaluate(self.activations[self.L-1],Y_train)
            grads = self.backward_propagatation(self.activations[self.L-1],Y_train)

            self.update_params(grads, alpha)

            print("{a} : {b}".format(a=i, b=cost))


    def evaluate(self, Y_nn, Y):
        m = Y.shape[1]
        cost = -1*np.sum(np.multiply(Y, np.log(Y_nn))) / m
        #cost = -1 * np.sum(np.multiply(self.Y, np.log(self.a2)) + 
        #            np.multiply(1-self.Y, np.log(1-self.a2)))
        #cost = cost / self.Y.shape[1]

        return np.squeeze(cost)

    def update_params(self, grads, alpha):
        for l in range(1, self.L):
            self.params['W'+str(l)] = self.params['W'+str(l)] - alpha*grads['dW'+str(l)]
            self.params['b'+str(l)] = self.params['b'+str(l)] - alpha*grads['db'+str(l)]

    def predict(self, X_test):
        a2 = self.forward_propagation(X_test)
        predictions = np.argmax(a2, axis=0)
        #predictions = np.array(a2>0.5).astype(int)
        return predictions

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def relu(self,z):
        return np.maximum(0,z)


