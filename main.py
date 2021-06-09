import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import csv


x=[]
y=[]
x_plot=[]
y_plot=[]
y_initial_values = []
data = []
# load data into the script
with open('./resources/tensorflow_tensorflowRegression.txt', 'r') as file:
    reader = csv.reader(file)
    counter = 0
    summ = 0
    c=0
    for row in reader:
        counter = counter + 1
        summ = summ + int(row[1])
        # 28 below is the number of days for which you want to have a sum of the commited lines
        if counter % 28 == 0:
            data.append([c,summ])
            x_plot.append(c)
            y_initial_values.append(summ)
            summ = 0
            c = c + 1

# shuffle points in order to choose the training set randomly, useful if the size of training set is smaller than dataset
rng = np.random.RandomState()
rng.shuffle(data)
data = data[:72]
# some preparation of data
for i in range (0,len(data)):
    data[i][0] = int (data[i][0])
    data[i][1] = int (data[i][1])
data = np.array(data)

data = data[data[:, 0].argsort()]

x = np.array([])
y = np.array([])
for i in data:
    x = np.append(x, int(i[0]))
    y = np.append(y, int(i[1]))

x_plot = np.array(x_plot)
y_initial_values = np.array(y_initial_values)


# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

# declare colors
colors = ['teal', 'red', 'gold','green','red','purple']
lw = 2

plt.scatter(x_plot,y_initial_values, color='green', label="initial points")

# Below in the loop description, after enumerate, insert the degrees, for which you want to generate the polynomial
for count, degree in enumerate([14]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    print("degree "+ str(degree) + " R2=" + str(r2_score(y_initial_values,y_plot)))
    print("degree "+ str(degree) + " MSE=" + str(mean_squared_error(y_initial_values, y_plot)))
    print("degree " + str(degree) + " MAE=" + str(mean_absolute_error(y_initial_values, y_plot)))
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="d %d" % degree)

plt.legend(loc='upper left')

plt.show()
