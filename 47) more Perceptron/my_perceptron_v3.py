import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib import gridspec

class PerceptronAnimation:

    def __init__(self, learning_rate_w, learning_rate_b, activation_method='sigmoid'):
        # self.w = np.random.rand()
        self.b = np.random.rand()
        self.lrw = learning_rate_w
        self.lrb = learning_rate_b
        self.method = activation_method
        self.fig = plt.figure(figsize=(15,5))
        
    def activation_func(self, x):
        if self.method == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if self.method == 'relu':
            return max([0, x])
        if self.method == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        if self.method == 'linear':
            return x
            
    def fit_and_eval(self, x_train, y_train,  epochs):
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1,1))
        self.w = np.random.rand(x_train.shape[1])
        spec = gridspec.GridSpec(1, 3, width_ratios=[2,1,1])
        self.ax1 = self.fig.add_subplot(spec[0])
        self.ax2 = self.fig.add_subplot(spec[1])
        self.ax3 = self.fig.add_subplot(spec[2])
        accs_train = []
        accs_test = []
        losses_train = []
        losses_test = []
        for ep in range(epochs):
            for i in range(len(x_train)):
                error = y_train[i] - self.activation_func(x_train[i] @ self.w + self.b)
                self.w += x_train[i] * error * self.lrw
                self.b += error * self.lrb

                self.ax1.clear()
                if len(self.w) == 1:
                    self.ax1.scatter(x_train, y_train)
                    self.ax1.plot(x_train, x_train * self.w + self.b, color='orange')

                if len(self.w) == 2:
                    x0, x1 = np.meshgrid(x_train[:,0], x_train[:,1])
                    y = x0 * self.w[0] + x1 * self.w[1] + self.b
                    self.ax1.scatter(x_train[:,0], x_train[:,1], y_train, alpha=.5)
                    self.ax1.plot_surface(x0, x1, y, color='red', alpha=.1)

                acc, loss = self.evaluate(x_train, y_train)
                accs_train.append(acc)
                losses_train.append(loss)
                    
                self.ax2.plot(accs_train, color='blue', label='train')
                # self.ax2.plot(accs_test, label='test')
                self.ax2.set_title('model accuracy')
                self.ax2.set_xlabel('iteration')
                # self.ax2.set_ylabel('accuracy')
                # self.ax2.legend()

                self.ax3.plot(losses_train, color='blue', label='train')
                # self.ax3.plot(losses_test, label='test')
                self.ax3.set_title('model loss')
                self.ax3.set_xlabel('iteration')
                # self.ax3.set_ylabel('loss')
                # self.ax3.legend()

    def predict(self, x_test):
            y_pred = np.zeros(x_test.shape[0])
            if len(self.w) == 1:
                y_pred = x_test * self.w
            else:
                for i in range(len(self.w)):
                    y_pred += x_test[:,i] * self.w[i]
            y_pred += self.b
            return np.array([self.activation_func(y) for y in y_pred])

    def calc_accuracy(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return r2_score(y_test, y_pred)

    def calc_loss(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean((y_pred - y_test) ** 2)

    def evaluate(self, x_test, y_test):
        return (self.calc_accuracy(x_test, y_test), self.calc_loss(x_test, y_test))

    def get_weights(self):
        return np.append(self.w, self.b)
