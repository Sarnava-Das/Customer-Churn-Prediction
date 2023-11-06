import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import importlib.util

# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))

# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def gb_model(X_train,y_train,X_test,y_test):
    # Create and train the Gradient Boosting model
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = gb_model.predict(X_test)

    # Calculate accuracy 
    accuracy = accuracy_score(y_test, predictions)

    # Evaluate the model
    print(f"Gradient Boost Accuracy: {accuracy * 100:.2f}%")

def rf_model(X_train,y_train,X_test,y_test):
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0) 

    # Fit the classifier to the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    rf_predictions = rf_classifier.predict(X_test)

    # Calculate and print accuracy
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

def lr_model(X_train,y_train,X_test,y_test):
   
    # Evaluate Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    predictions = lr.predict(X_test)

    # Calculate accuracy 
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

def getfile():
    path=[]
    for dirname, _, filenames in os.walk(source_file.ROOT_DIR): 
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    for filename in path:
        if(os.path.basename(filename)==source_file.TRAIN_SET_PROCESSED_NAME): 
            train_set_filename=filename
       
    return train_set_filename

def main():
    train_set_file=getfile()
    X=pd.read_csv(train_set_file)
    
    # Target variable
    y = X['Exited']

    # Exclude the 'Exited' column from the features to select relevant features
    X = X.drop(columns=['Exited'])
    
    # Split the data into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    lr_model(X_train,y_train,X_test,y_test)
    gb_model(X_train,y_train,X_test,y_test)
    rf_model(X_train,y_train,X_test,y_test)

if __name__ == "__main__":
    main()