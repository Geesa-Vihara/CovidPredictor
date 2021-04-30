from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS, cross_origin
from twilio.rest import Client
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()
env_path = Path('.')/'.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():    
    request_data = request.get_json()
    test_date = request_data['test_date']
    cough = request_data['cough']
    fever = request_data['fever']
    sore_throat = request_data['sore_throat']    
    shortness_of_breath = request_data['shortness_of_breath']
    head_ache = request_data['head_ache']
    age_60_and_above = request_data['age_60_and_above']
    gender = request_data['gender']   
    test_indication = request_data['test_indication'] 
    try:
        prediction = preprocessDataAndPredict(cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, gender, test_indication)
        print(prediction)
        prediction = "positive" if prediction[0] == 1 else "negative"
        return jsonify({"prediction":prediction}), 200   
    except ValueError:
        return "Please Enter valid values"
    pass 

def preprocessDataAndPredict(cough, fever, sore_throat, shortness_of_breath, head_ache, age_60_and_above, gender, test_indication):
    age_60_and_above = "1" if age_60_and_above == "Yes" else "0"
    gender = "1" if gender == "female" else "0"  
    test_indication = "1" if test_indication == 'Contact with confirmed' else "2" if test_indication == 'Abroad' else "0" 
    test_data = [int(cough), int(fever), int(sore_throat), int(shortness_of_breath), int(head_ache), int(age_60_and_above), int(gender), int(test_indication)]  
    test_data = np.array(test_data)    
    test_data = test_data.reshape(1,-1)    
    file = open('output/covid-predictor_model.pkl','rb')    
    trained_model = joblib.load(file)
    prediction = trained_model.predict(test_data)  
    return prediction
    pass

@app.route('/sms', methods=['POST'])
@cross_origin()
def sms():
    request_data = request.get_json()
    from_number = request_data['from_number']
    to_number = request_data['to_number']
    message_body = request_data['message_body']
    account_sid = os.getenv("ACCOUNT_SID")
    auth_token = os.getenv("AUTH_TOKEN")
    client = Client(account_sid, auth_token)
    message = client.messages.create(
                body=message_body,
                from_=from_number,
                to=to_number
            )
    return message.sid

@app.route('/')
@cross_origin()
def index():
    return "<h1>COVID-19 PREDICTOR!</h1>"

if __name__ == '__main__':
    app.run(threaded=True, port=5000)