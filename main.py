# Import necessary libraries
import pickle
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
import pandas as pd


app = Flask(__name__, template_folder='templates')

# Load the pickled model
with open('D:/Projects/Customer-Churn-Prediction/models/linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Open the file in binary read mode and load the label encoder
with open('D:/Projects/Customer-Churn-Prediction/models/label_encoding.pkl', 'rb') as file:
    loaded_label_encoder = pickle.load(file)


# Define a function to preprocess the input and make predictions
def predict_genre(df):
    
    

   
    # predicted_genre = model.predict(credit_score,location,gender,age,tenure,balance,no_of_products,has_credit_card,active_member,est_salary)
    predicted_genre = model.predict(df)
    
    return predicted_genre


# Define a route to handle the HTML form
@app.route('/', methods=['GET', 'POST'])
def predict_movie_genre():
    if request.method == 'POST':
        data = {
         'CreditScore': [int(request.form['credit_score'])],
            'Geography': [request.form['location']],
            'Gender': [request.form['gender']],
            'Age': [int(request.form['age'])],
            'Tenure': [int(request.form['tenure'])],
            'Balance': [float(request.form['balance'])],
            'NumOfProducts': [int(request.form['no_of_products'])],
            'HasCrCard': [int(request.form['has_credit_card'])],
            'IsActiveMember': [int(request.form['active_member'])],
            'EstimatedSalary': [float(request.form['est_salary'])]
        }
        df = pd.DataFrame(data)
        # location = request.form['location']
        # gender = request.form['gender']
        # age = request.form['age']
        # credit_score = request.form['credit_score']
        # has_credit_card = request.form['has_credit_card']
        # balance = request.form['balance']
        # est_salary = request.form['est_salary']
        # active_member = request.form['active_member']
        # tenure = request.form['tenure']
        # no_of_products = request.form['no_of_products']
       
        # predicted_genre = predict_genre(location,gender,age,credit_score,has_credit_card,balance,est_salary,active_member,tenure,no_of_products)
        predicted_genre = predict_genre(df)
        # print(predicted_genre)
        return render_template('index.html', genres=predicted_genre)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    # predict_movie_genre()
    app.run()
