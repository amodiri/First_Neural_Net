import numpy as np
import mnist_loader
import sys
import numpy
from sklearn.preprocessing import OneHotEncoder
numpy.set_printoptions(threshold=sys.maxsize)

DEBUG=True
class Neural_Net(object):

    def log(self, msg):
        if (DEBUG):
            print(msg)

    def __init__(self, nn_dims, init_method='init_with_rands'):
        self.L = len(nn_dims)
        self.params = {}
        self.activations = {}
        self.Z = {}
        self.init_with_rands(nn_dims)

    def init_with_zeroBias(self):
        pass

    def init_with_rands(self, nn_dims):
        for l in range(1,self.L):
            self.params['W'+str(l)] = np.random.randn(nn_dims[l],nn_dims[l-1]) * 0.01
            self.params['b'+str(l)] = np.zeros((nn_dims[l],1))

    def init_better(self):
        self.w1 = np.zeros((nh,nx))

    def forward_propagation(self, X):
        self.activations[0] = X
        L = self.L - 1
        for l in range(1, L):
            self.Z[l] = np.dot(self.params['W'+str(l)], self.activations[l-1]) + self.params['b'+str(l)]
            self.activations[l] = self.relu(self.Z[l])


        self.Z[L] = np.dot(self.params['W'+str(L)], self.activations[L-1]) + self.params['b'+str(L)]
        temp = np.exp(self.Z[L])
        self.activations[L] = temp / np.sum(temp, axis=0)

    def backward_propagatation(self, Y_nn, Y):
        L = self.L - 1
        grads = {}
        m = Y.shape[1]
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
        return np.squeeze(cost)

    def update_params(self, grads, alpha):
        for l in range(1, self.L):
            self.params['W'+str(l)] = self.params['W'+str(l)] - alpha*grads['dW'+str(l)]
            self.params['b'+str(l)] = self.params['b'+str(l)] - alpha*grads['db'+str(l)]

    def predict(self, X_test, Y_test):
        m = X_test.shape[1]
        self.forward_propagation(X_test)
        Y_pred = self.activations[self.L-1]
        Y_pred = np.argmax(Y_pred, axis=0)
        Y_pred = Y_pred.reshape(1, Y_pred.shape[0])
        accuracy = np.sum((Y_pred==Y_test)/m)
        print("Accuracy : ", accuracy*100, " %")
        return Y_pred

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def relu(self,z):
        return np.maximum(0,z)


