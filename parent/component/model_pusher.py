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
    with open(source_file.PRED_MODEL_PATH, 'wb') as file:
        pickle.dump(best_rf_model, file)

   

def getfile():
    path=[]
    for dirname, _, filenames in os.walk(source_file.ROOT_DIR): 
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    encoder_file=""
    for filename in path:
        if(os.path.basename(filename)==source_file.TRAIN_SET_PROCESSED_NAME): 
            train_set_filename=filename
        if(os.path.basename(filename)==source_file.TRAIN_SET):
            encoder_file=filename  
    return train_set_filename,encoder_file

def main():
    train_set_file,encoder_file_name=getfile()
    encoded_csv=pd.read_csv(encoder_file_name)   
    
    # Initialize separate label encoders for Geography and Gender
    geography_label_encoder = LabelEncoder()
    gender_label_encoder = LabelEncoder()

    # Fit and transform the labels for Geography and Gender
    geography_label_encoder.fit_transform(encoded_csv[source_file.COLUMN1_ENCODE])
    gender_label_encoder.fit_transform(encoded_csv[source_file.COLUMN2_ENCODE])

    # Save the label encoders in a dictionary
    label_encoders = {
        'Geography': geography_label_encoder,
        'Gender': gender_label_encoder
    }

    # Open the file in binary write mode and save the label encoder
    with open(source_file.ENCODING_PATH, 'wb') as file:
        pickle.dump(label_encoders, file)

    X=pd.read_csv(train_set_file)   
    # Target variable
    y = X['Exited']
    
    # Exclude the 'Exited' column from the features to select relevant features
    X = X.drop(columns=['Exited'])
    
    # Split the data into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    params=tune_model(X_train,y_train)
    train_model(params,X_train,y_train,X_test,y_test)
   
if __name__ == "__main__":
    main()
