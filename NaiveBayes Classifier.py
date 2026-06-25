from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_pred,y_test))
print("\nClassification Report:\n",classification_report(y_pred,y_test))