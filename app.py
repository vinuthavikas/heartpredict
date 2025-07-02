from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    features = [float(request.form[f]) for f in [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
    
    # Make prediction
    prediction = model.predict([features])[0]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
