# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Leave Forecast
@app.route('/predict-leave-trends', methods=['POST'])
def predict_leave_trends():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df.rename(columns={'leave_date': 'ds', 'leave_count': 'y'}, inplace=True)

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        result = forecast[['ds', 'yhat']].tail(30).to_dict(orient='records')
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. Absenteeism Risk Prediction
@app.route('/predict-absenteeism', methods=['POST'])
def predict_absenteeism():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df['is_frequent_late'] = df['late'] > 180
        df['low_hours'] = df['hours_worked'] < 4
        df['risk'] = (df['is_frequent_late'] & df['low_hours']).astype(int)

        return jsonify(df[['employee_id', 'risk']].to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 3. Fraud Detection
@app.route('/detect-leave-fraud', methods=['POST'])
def detect_fraud():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df['start_day'] = pd.to_datetime(df['start_date']).dt.dayofweek
        df['end_day'] = pd.to_datetime(df['end_date']).dt.dayofweek
        df['fraud_flag'] = ((df['start_day'] == 4) | (df['end_day'] == 0)).astype(int)

        return jsonify(df[['employee_id', 'start_date', 'end_date', 'fraud_flag']].to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
