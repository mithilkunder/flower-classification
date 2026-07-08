import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

# Load Dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("First 5 Rows")
print(X.head())

print("\nDataset Info")
print(X.info())

print("\nStatistics")
print(X.describe())

# Pair Plot
sns.pairplot(X)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap="Blues")
plt.show()

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLogistic Regression Accuracy")
print(accuracy_score(y_test, lr_pred))

print(confusion_matrix(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy")
print(accuracy_score(y_test, dt_pred))

print(confusion_matrix(y_test, dt_pred))

print(classification_report(y_test, dt_pred))

# Save Better Model
if accuracy_score(y_test, lr_pred) >= accuracy_score(y_test, dt_pred):
    joblib.dump(lr, "iris_model.pkl")
    print("Logistic Regression model saved.")
else:
    joblib.dump(dt, "iris_model.pkl")
    print("Decision Tree model saved.")