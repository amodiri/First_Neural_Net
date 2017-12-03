import numpy as np

class Neural_Net(object):

    def __init__(self, X_train, Y_train, nh, init_method='init_with_rands'):
        self.nx = X_train.shape[0]
        self.nh = nh
        self.ny = Y_train.shape[0]
        self.X = X_train
        self.Y = Y_train
        #temporary
        self.init_with_rands()

    def init_with_zeroBias(self):
        self.w1 = np.random.randn(self.nh,self.nx) * 0.01
        self.b1 = np.zeros((self.nh, 1))
        self.w2 = np.random.randn(self.ny,self.nh) * 0.01
        self.b2 = np.zeros((self.ny,1))

    def init_with_rands(self):
        self.w1 = np.random.randn(self.nh,self.nx) * 0.01
        self.b1 = np.random.randn(self.nh, 1)
        self.w2 = np.random.randn(self.ny,self.nh) * 0.01
        self.b2 = np.random.randn(self.ny,1)

    def init_better(self):
        self.w1 = np.zeros((nh,nx))

    def forward_propagation(self, X_test='None'):
        #for b,w in zip(self.Bs, self.Ws):
        if (X_test):
            a1 = self.sigmoid(np.dot(self.w1, X_test) + self.b1)
            a2 = self.sigmoid(np.dot(self.w2, self.a1) + self.b2)
            return a2
        else:
            self.a1 = self.sigmoid(np.dot(self.w1, self.X) + self.b1)
            self.a2 = self.sigmoid(np.dot(self.w2, self.a1) + self.b2)
            print("a1 : ", self.a1.shape)
            print("a2 : ", self.a2.shape)

    def backward_propagatation(self):
        m = X.shape[1]
        dz2 = self.a2 - self.Y /m
        dz1 = np.multiply(np.dot(self.w2.T, dz2), 1-np.power(self.a1,2))

        self.dw2 = np.dot(dz2, self.a1.T)
        self.db2 = np.sum(dz2, axis=1, keepdims=True)
        self.dw1 = np.dot(dz1, self.X.T) / m
        self.db1 = np.sum(dz1, axis=1, keepdims=True)

    def nn_training(self, iterations, alpha):
        for i in range(0, iterations):
            self.forward_propagation()
            self.backward_propagatation()

            self.w1 = w1 - alpha * self.dw1
            self.b1 = b1 - alpha * self.db1
            self.w2 = w2 - alpha * self.dw2
            self.b2 = b2 - alpha * self.db2

            cost = self.evaluate()
        trained_param = {"w1" : self.w1,
                         "w2" : self.w2,
                         "b1" : self.b1,
                         "b2" : self.b2}
        return trained_param

    def evaluate(self):
        cost = -1 * np.sum(np.multiply(self.Y, np.log(self.a2)) + 
                    np.multiply(1-self.Y, np.log(1-self.a2)))
        cost = cost / self.Y.shape[1]
        return cost

    def predict(self, X_test):
        a2 = forward_propagation(X_test)
        predictions = np.array(a2>0.5).astype(int)
        return predictions

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
