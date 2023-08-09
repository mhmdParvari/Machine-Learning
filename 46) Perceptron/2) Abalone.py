import pandas as pd
from sklearn.model_selection import train_test_split
from my_perceptron import PerceptronAnimation

data = pd.read_csv('datasets/abalone_csv.csv')
x = data['Diameter'].to_numpy()
y = data['Shell_weight'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.9)

p = PerceptronAnimation(.2, .1, 5)
p.fit(x_train, y_train)
