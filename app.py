# Heart Disease Prediction - Single File Project

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------- DATA LOADING --------------------
# Dataset should be in the same folder as this file
DATA_PATH = "heart.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# -------------------- PREPROCESSING --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- TRAIN TEST SPLIT --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------- MODEL TRAINING --------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------- MODEL EVALUATION --------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -------------------- SAVE MODEL --------------------
joblib.dump(model, "heart_model.pkl")

# -------------------- FLASK APPLICATION --------------------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction</title>
</head>
<body>
    <h2>Heart Disease Prediction</h2>
    <form method="post">
        <input type="number" step="any" name="age" placeholder="Age" required><br>
        <input type="number" step="any" name="sex" placeholder="Sex" required><br>
        <input type="number" step="any" name="cp" placeholder="Chest Pain" required><br>
        <input type="number" step="any" name="trestbps" placeholder="Resting BP" required><br>
        <input type="number" step="any" name="chol" placeholder="Cholesterol" required><br>
        <input type="number" step="any" name="fbs" placeholder="Fasting Blood Sugar" required><br>
        <input type="number" step="any" name="restecg" placeholder="Rest ECG" required><br>
        <input type="number" step="any" name="thalach" placeholder="Max Heart Rate" required><br>
        <input type="number" step="any" name="exang" placeholder="Exercise Angina" required><br>
        <input type="number" step="any" name="oldpeak" placeholder="Oldpeak" required><br>
        <input type="number" step="any" name="slope" placeholder="Slope" required><br>
        <input type="number" step="any" name="ca" placeholder="CA" required><br>
        <input type="number" step="any" name="thal" placeholder="Thal" required><br>
        <br>
        <button type="submit">Predict</button>
    </form>

    {% if result is not none %}
        <h3>Result: {{ result }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(features_scaled)[0]

        result = "Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease"

    return render_template_string(HTML_TEMPLATE, result=result)

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)
