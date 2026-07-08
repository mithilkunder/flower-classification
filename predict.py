import joblib

model = joblib.load("iris_model.pkl")

print("Enter Flower Measurements")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

prediction = model.predict([[sl, sw, pl, pw]])

flowers = [
    "Setosa",
    "Versicolor",
    "Virginica",
]

print("\nPredicted Flower:", flowers[prediction[0]])