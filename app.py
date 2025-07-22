from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define feature list
features = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in features]
        scaled_data = scaler.transform([input_data])
        prediction = model.predict(scaled_data)
        result = "Patient is likely to survive" if prediction[0] == 0 else "High risk of death"
        return f"<h1>Prediction: {result}</h1>"
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
