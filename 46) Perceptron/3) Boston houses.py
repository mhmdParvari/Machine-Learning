import pandas as pd
from sklearn.model_selection import train_test_split
from my_perceptron import PerceptronAnimation

data = pd.read_csv('datasets/BostonHousing.csv')
x = data[['indus', 'age']].to_numpy()
y = data['nox'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

print(y_train.shape)
p = PerceptronAnimation(.00001, .0001, 5)
p.fit(x_train, y_train)