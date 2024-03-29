import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn import neighbors

# Import data from .csv and split into features and labels
Z = np.genfromtxt('data.csv', delimiter=',')
X, y = Z[:,:-1], Z[:,-1]

# Print function for scatter plot
def print_plot(X, y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

print_plot(X, y)

# Calculate and plot generalization error by faetures and labels
def show_generalization_error(X_param, y_param):
    kfold = KFold(10)
    kfold.get_n_splits(X_param)

    score_map = []

    for k in range(1, 180, 2):
        scores = []
        neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
        for train_index, test_index in kfold.split(X_param):
            X_train, X_test = X_param[train_index], X_param[test_index]
            y_train, y_test = y_param[train_index], y_param[test_index]
            neigh.fit(X_train, y_train)
            pred = neigh.predict(X_test)
            np.mean(pred != y_test)
            scores.append(np.mean(pred != y_test))
        score_map.append([k, np.mean(scores)])

    score_array = np.array(score_map)
    plt.scatter(score_array[:,0], score_array[:,1])
    plt.xlabel('k')
    plt.ylabel('prediction error rate')
    plt.show()

show_generalization_error(X, y)


# Add noise to data set: probability 1 out of 5
np.random.seed(1234)
y_random = np.empty(len(y))

for i in range(0, len(y), 1):
    y_random[i] = y[i]
    if np.random.rand() <= 0.2:
            y_random[i] = y[i] * -1

print_plot(X, y_random)
show_generalization_error(X, y_random)

# Add dimensions
np.random.seed(1234)

for f in range(1,5,1):
    random_column = np.random.rand(len(X), f)
    Xnd = np.hstack((X, random_column))
    show_generalization_error(Xnd, y)
