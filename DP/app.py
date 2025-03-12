# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# Feature columns
X = df.drop("Outcome", axis=1)

# Target column
y = df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict function
def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Web application routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = {
            "Pregnancies": int(request.form['pregnancies']),
            "Glucose": int(request.form['glucose']),
            "BloodPressure": int(request.form['blood_pressure']),
            "SkinThickness": int(request.form['skin_thickness']),
            "Insulin": int(request.form['insulin']),
            "BMI": float(request.form['bmi']),
            "DiabetesPedigreeFunction": float(request.form['dpf']),
            "Age": int(request.form['age'])
        }

        result = predict_diabetes(input_data)
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
