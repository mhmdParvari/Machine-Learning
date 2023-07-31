import numpy as np
from numpy.linalg import inv

class LinearLeastSquares:
    
    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        self.m = self.calculate_slope()

    def calculate_slope(self):
        if len(self.x.shape) == 1:
            x = self.x.reshape((len(self.x),1))
        else:
            x = self.x
            
        m = np.matmul(inv(np.matmul(x.T, x)), np.matmul(x.T, self.y))
        return m

    def predict(self, x_test):
        if len(self.m) == 1:
            return x_test * self.m

        y = np.zeros(len(x_test))
        for i in range(len(self.m)):
            y += x_test[:, i] * self.m[i]
        return y

    def get_slope(self):
        return self.m