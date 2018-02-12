"""
This module implements neural network models with arbitrary number of layers
and hidden units. The size of network should be defined when the object is
created. Forward propagation uses Gradient descent algorithm and there are
several options available for initializing the Neural Net parameters.
"""

import numpy as np
import mnist_loader
import sys
import numpy
from sklearn.preprocessing import OneHotEncoder
numpy.set_printoptions(threshold=sys.maxsize)


class Neural_Net(object):

    def __init__(self, nn_dims, init_method='init_with_rands'):
        """
        Initializes the class variables and calls the selected initialization
        method for NN parameters.

        Args:
            nn_dims (list): size of the neural net
            init_method (str): method of parameters initialization. If not
            specified, random initialization will be selected.

        Returns:
            None
        """
        self.L = len(nn_dims)
        self.params = {}
        self.activations = {}
        self.Z = {}

        if(init_method == 'init_with_rands'):
            self.init_with_rands(nn_dims)
        elif(init_method == 'init_with_he'):
            self.init_with_he(nn_dims)


    def init_with_rands(self, nn_dims):
        """
        Initializes the parameters of NN randomly.

        Args:
            nn_dims (list): size of the neural net

        Returns:
            None
        """
        for l in range(1,self.L):
            self.params['W'+str(l)] = np.random.randn(nn_dims[l],nn_dims[l-1]) * 0.01
            self.params['b'+str(l)] = np.zeros((nn_dims[l],1))


    def init_with_he(self, nn_dims):
        """
        Initializes the parameters of NN with the method introduced by He et al

        Args:
            nn_dims (list): size of the neural net

        Returns:
            None
        """
        for l in range(1,self.L):
            self.params['W'+str(l)] = np.random.randn(nn_dims[l],nn_dims[l-1]) * np.sqrt(2/nn_dims[l-1])
            self.params['b'+str(l)] = np.zeros((nn_dims[l],1))


    def forward_propagation(self, X):
        """
        Implements forward propagation using gradient descent algorithm.

        Args:
            X (array): Input of NN

        Returns:
            None
        """
        self.activations[0] = X
        L = self.L - 1
        for l in range(1, L):
            self.Z[l] = np.dot(self.params['W'+str(l)], self.activations[l-1]) + self.params['b'+str(l)]
            self.activations[l] = self.relu(self.Z[l])


        self.Z[L] = np.dot(self.params['W'+str(L)], self.activations[L-1]) + self.params['b'+str(L)]
        temp = np.exp(self.Z[L])
        self.activations[L] = temp / np.sum(temp, axis=0)


    def backward_propagatation(self, Y_nn, Y):
        """
        Implements backward propagation

        Args:
            Y_nn (array): NN estimated output
            Y (array): the given output

        Returns:
            Dictionary containing all gradient descent updates for parameters
            of NN
        """
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


    def nn_training(self, X_train, Y_train, iterations, alpha, lambd):
        """
        Implements backward propagation

        Args:
            X_train (array): training set inputs
            Y_train (array): training set outputs
            iterations (int): number of iterations for gradient descent
            alpha (float): learning rate
            lambd (float): Regularization parameter

        Returns:
            Dictionary containing all gradient descent updates for parameters
            of NN
        """
        for i in range(0, iterations):
            self.forward_propagation(X_train)
            cost = self.evaluate(self.activations[self.L-1],Y_train, lambd)
            grads = self.backward_propagatation(self.activations[self.L-1],Y_train)

            self.update_params(grads, alpha, lambd, Y_train.shape[1])

            print("{a} : {b}".format(a=i, b=cost))


    def evaluate(self, Y_nn, Y, lambd):
        """
        Computes the cost value for the trained NN

        Args:
            Y_nn (array): NN estimated output
            Y (array): the given output
            lambd (float): Regularization parameter

        Returns:
            value of cost function based on the given output values and their
            estimations
        """
        m = Y.shape[1]
        cost = -1*np.sum(np.multiply(Y, np.log(Y_nn))) / m

        reg = 0
        for l in range(1, self.L):
            reg += np.sum(np.square(self.params['W'+str(l)]))
        cost_with_reg = cost + reg * lambd / (2*m)

        return np.squeeze(cost_with_reg)


    def update_params(self, grads, alpha, lambd, m):
        """
        Updates the parameters of NN based on the learning rate and the update
        values

        Args:
            grads (dict): all gradient descent updates for parameters of NN
            alpha (float): learning rate
            lambd (float): Regularization parameter
            m (int): number of training data

        Returns:
            None
        """
        for l in range(1, self.L):
            self.params['W'+str(l)] = self.params['W'+str(l)] - alpha*grads['dW'+str(l)] - lambd * self.params['W'+str(l)] / m
            self.params['b'+str(l)] = self.params['b'+str(l)] - alpha*grads['db'+str(l)]


    def predict(self, X_test, Y_test):
        """
        Predicts the output based on the given input. Also, calculated the accuracy of the prediction.

        Args:
            X_test (array): test set inputs
            Y_test (array): test set outputs

        Returns:
            An array of the predicted outputs
        """
        m = X_test.shape[1]
        self.forward_propagation(X_test)
        Y_pred = self.activations[self.L-1]
        Y_pred = np.argmax(Y_pred, axis=0)
        Y_pred = Y_pred.reshape(1, Y_pred.shape[0])
        accuracy = np.sum((Y_pred==Y_test)/m)
        print("Accuracy : ", accuracy*100, " %")
        return Y_pred


    def sigmoid(self, z):
        """
        Calculates the value of Sigmoid function for the given input
        """
        return 1/(1+np.exp(-z))


    def relu(self,z):
        """
        Calculates the value of Relu function for the given input
        """
        return np.maximum(0,z)


