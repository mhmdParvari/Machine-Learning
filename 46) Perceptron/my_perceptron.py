import numpy as np
import matplotlib.pyplot as plt

class PerceptronAnimation:

    def __init__(self, learning_rate_w, learning_rate_b, epochs):
        # self.w = np.random.rand()
        self.b = np.random.rand()
        self.lrw = learning_rate_w
        self.lrb = learning_rate_b
        self.epochs = epochs
        self.fig = plt.figure()
        
    def fit(self, x_train, y_train):
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1,1))
            
        self.w = np.random.rand(x_train.shape[1])
        if len(self.w) == 1:
            self.ax1 = self.fig.add_subplot(1,2,1)
        else:
            self.ax1 = self.fig.add_subplot(1,2,1, projection='3d')
        self.ax2 = self.fig.add_subplot(1,2,2)
        losses = []
        for ep in range(self.epochs):
            for i in range(len(x_train)):
                error = y_train[i] - x_train[i] @ self.w - self.b  # wasted +2 hours to find the problem just to find out my dumb brain had put +b
                losses.append(error ** 2)
                self.w += x_train[i] * error * self.lrw
                self.b += error * self.lrb

                self.ax1.clear()
                if len(self.w) == 1:
                    self.ax1.scatter(x_train, y_train)
                    self.ax1.plot(x_train, x_train * self.w + self.b, color='red')

                if len(self.w) == 2:
                    x0, x1 = np.meshgrid(x_train[:,0], x_train[:,1])
                    y = x0 * self.w[0] + x1 * self.w[1] + self.b
                    self.ax1.scatter(x_train[:,0], x_train[:,1], y_train, alpha=.5)
                    self.ax1.plot_surface(x0, x1, y, color='red', alpha=.1)

                self.ax2.clear()
                self.ax2.plot(losses)
                self.ax2.set_title('Loss')
                plt.pause(.05)

    def predict(self, x_test):
        return x_test * self.w + self.b

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean((y_test - y_pred) ** 2)
