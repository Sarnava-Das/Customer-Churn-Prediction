import os
import pandas as pd
import pathlib 




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import pickle
from model_trainer import tune_model
import importlib.util


# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def train_model(best_params,X_train,y_train,X_test,y_test):
    # Train the model with the best hyperparameters
    best_rf_model = RandomForestClassifier(random_state=42, **best_params)
    best_rf_model.fit(X_train, y_train)
     # Save the trained model to a pickle file
    with open('D:/Projects/Customer-Churn-Prediction/models/linear_regression_model.pkl', 'wb') as file:
        pickle.dump(best_rf_model, file)

   

def getfile():
    path=[]
    for dirname, _, filenames in os.walk('D:/Projects/Customer-Churn-Prediction'): #'Projects' is the folder name in which the required files are saved
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    for filename in path:
        if(os.path.basename(filename)=='train_processed.csv'): #filename with extension
            train_set_filename=filename
    return train_set_filename

def main():
    train_set_file=getfile()
      
    
    # Select relevant features
    X=pd.read_csv(train_set_file)
    # Target variable
    y = X['Exited']
    
    # Exclude the 'Exited' column from the features
    X = X.drop(columns=['Exited'])
    
    # Split the data into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
 
    # Initialize a label encoder
    label_encoder = LabelEncoder()

    # Fit the encoder and transform the labels
    label_encoder.fit_transform(X_train['Geography'])
    label_encoder.fit_transform(X_train['Gender'])
    label_encoder.fit_transform(X_test['Geography'])
    label_encoder.fit_transform(X_test['Gender'])
    
    # Open the file in binary write mode and save the label encoder
    with open('D:/Projects/Customer-Churn-Prediction/models/label_encoding.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    params=tune_model(X_train,y_train)
    train_model(params,X_train,y_train,X_test,y_test)
   
if __name__ == "__main__":
    main()
