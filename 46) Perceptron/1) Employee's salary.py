import numpy as np
from sklearn import datasets
from my_perceptron import PerceptronAnimation
import matplotlib.pyplot as plt


x, y = datasets.make_regression(100, n_features=1, noise=15)
x = np.interp(x, (x.min(),x.max()), (0,30))
y = np.interp(y, (y.min(),y.max()), (5,25))

p = PerceptronAnimation(0.0001, .001, 5)
p.fit(x,y)
