from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
neighbors = np.arange(1, 9)

train_accuracy = []
test_accuracy = []

for k in neighbors:

    model = KNeighborsClassifier(k)

    model.fit(X_train, y_train)

  
    train_accuracy.append(model.score(X_train, y_train))
  
    test_accuracy.append(model.score(X_test, y_test))


plt.plot(neighbors, test_accuracy,label="Testing dataset Accuracy")
plt.plot(neighbors, train_accuracy,label="Training dataset Accuracy")

plt.xlabel("neighbors")
plt.ylabel("Accuracy")

plt.legend()
plt.show()