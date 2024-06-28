from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el pipeline
pipeline = joblib.load('lgbm_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    new_data = pd.DataFrame({
        'ProductCategory': [data['ProductCategory']],
        'ProductBrand': [data['ProductBrand']],
        'ProductPrice': [float(data['ProductPrice'])],
        'CustomerAge': [int(data['CustomerAge'])],
        'CustomerGender': [int(data['CustomerGender'])],
        'PurchaseFrequency': [int(data['PurchaseFrequency'])],
        'CustomerSatisfaction': [int(data['CustomerSatisfaction'])]
    })

    # Hacer una predicción
    prediction = pipeline.predict(new_data)
    result = 'Purchase Intent' if prediction[0] == 1 else 'No Purchase Intent'

    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
