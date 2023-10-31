# Import necessary libraries
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import importlib.util

source_file_path = "parent/constants/__init__.py"

spec = importlib.util.spec_from_file_location('__init__', source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load the pickled model
with open(source_file.PRED_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


# Open the file in binary read mode and load the label encoder
with open(source_file.ENCODING_PATH, 'rb') as file:
    loaded_label_encoder = pickle.load(file)


# function to make predictions
def predict_churn(df):
  
    predicted_churn = model.predict(df)
    if predicted_churn ==1:
        return 'Customer not retained'
    elif predicted_churn ==0:
        return 'Customer retained '

    
  
# Define a route to handle the HTML form
@app.route('/', methods=['GET', 'POST'])
def predict_customer_churn():
    if request.method == 'POST':
    
       
       
        encoded_location = loaded_label_encoder['Geography'].transform([request.form['location']])
        encoded_gender = loaded_label_encoder['Gender'].transform([request.form['gender']])

     
       
        if str.lower(request.form['has_credit_card']) == 'yes':
            credit_card =1
        elif str.lower(request.form['has_credit_card']) == 'no':
            credit_card =0
        
        if str.lower(request.form['active_member']) == 'yes':
            active_member =1
        elif str.lower(request.form['active_member']) == 'no':
            active_member =0

        data = {
         'CreditScore': [int(request.form['credit_score'])],
            'Geography': [encoded_location],
            'Gender': [encoded_gender],
            'Age': [int(request.form['age'])],
            'Tenure': [int(request.form['tenure'])],
            'Balance': [float(request.form['balance'])],
            'NumOfProducts': [int(request.form['no_of_products'])],
            'HasCrCard': [credit_card],
            'IsActiveMember': [active_member],
            'EstimatedSalary': [float(request.form['est_salary'])]
        }
        df = pd.DataFrame(data)
      
        predicted_churn = predict_churn(df)
      
        return render_template('index.html', churn=predicted_churn)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run()
