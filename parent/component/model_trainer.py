import os
import pandas as pd
import pathlib 

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



import importlib.util



# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))


# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)

def tune_model(X_train,y_train):
    
        # Define the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Define a grid of hyperparameters to search
    param_grid = {
        'n_estimators': [50, 100, 200],          # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],        # Maximum depth of individual trees
        'min_samples_split': [2, 5, 10],       # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]          # Minimum number of samples required to be a leaf node
    }

        # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)

    # Fit the random search to the data
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = random_search.best_params_
    return best_params

def train_model(best_params,X_train,y_train,X_test,y_test):
    # Train the model with the best hyperparameters
    best_rf_model = RandomForestClassifier(random_state=42, **best_params)
    best_rf_model.fit(X_train, y_train)
    # Make predictions
    predictions = best_rf_model.predict(X_test)
   
      # Calculate accuracy and report
    accuracy = accuracy_score(y_test, predictions)
    
    print("Best Hyperparameters:", best_params)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions))


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
    
 
    
    params=tune_model(X_train,y_train)
    train_model(params,X_train,y_train,X_test,y_test)
  


if __name__ == "__main__":
    main()
