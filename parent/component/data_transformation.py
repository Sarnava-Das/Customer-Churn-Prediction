from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import pathlib 
import importlib.util

# Specify the absolute path to source_file.py
source_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../constants/__init__.py'))

# Use importlib to import source_file
spec = importlib.util.spec_from_file_location("__init__", source_file_path)
source_file = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_file)


def process_data(data):
    # Imputations
    data = data.drop(columns=[source_file.COLUMN1_IMPUTE, source_file.COLUMN2_IMPUTE, source_file.COLUMN3_IMPUTE])

    # Encode categorical variables
    encoder = LabelEncoder()
    data[source_file.COLUMN1_ENCODE] = encoder.fit_transform(data[source_file.COLUMN1_ENCODE])
    data[source_file.COLUMN2_ENCODE] = encoder.fit_transform(data[source_file.COLUMN2_ENCODE])

    return data

def getfile():
    path=[]
    for dirname, _, filenames in os.walk('D:/Projects/Customer-Churn-Prediction'): #'Projects' is the folder name in which the required files are saved
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    for filename in path:
        if(os.path.basename(filename)=='Churn_Modelling.csv'): #filename with extension
            train_set_filename=filename
        
    return train_set_filename

def batch_processing(data,filename):
  
    batch_size = 1000  
    
    processed_data=pd.DataFrame()
    
    if filename=='Churn_Modelling':
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
        
            # Get the current batch of data
            batch_data = data.iloc[batch_start:batch_end]
            processed_data = pd.concat([processed_data, process_data(batch_data)])

   
    return processed_data
    """
    Dealing with Short Words or Acronyms
    Custom handling based on your specific needs.

    Handling Negations
    Custom handling based on your specific needs.

    Remove Rare Words or Low-Frequency Terms
    Custom handling based on your specific needs.

    Addressing Data Imbalance
    Use resampling techniques like oversampling (e.g., SMOTE) or undersampling.
  
def tf_idf(train_data, test_data, column_to_clean, feature_name_column):
    # Create a TF-IDF vectorizer
    max_features = 1000  # You can adjust this value as needed
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    # Create empty lists to store TF-IDF matrices and feature names
    tfidf_strings_train = []  # Store TF-IDF values as strings for training data
    tfidf_strings_test = []   # Store TF-IDF values as strings for test data
    feature_names_list = []

    for dataset, tfidf_strings in [(train_data, tfidf_strings_train), (test_data, tfidf_strings_test)]:
        for row_index, row in dataset.iterrows():
            # Fit and transform the text data for the current row using TF-IDF
            tfidf_matrix = tfidf_vectorizer.fit_transform([row[column_to_clean]])

            # Convert the TF-IDF values to a string using a delimiter (e.g., semicolon)
            # tfidf_values = ";".join(map(str, tfidf_matrix.toarray().flatten()))
            tfidf_values =  tfidf_matrix

            # Get the feature names (unique words) from the vectorizer
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Append the TF-IDF values and feature names to the respective lists
            tfidf_strings.append(tfidf_values)
            feature_names_list.append(feature_names)

    # Create new columns for the TF-IDF values as strings in both training and test dataframes
    train_data['TF-IDF'] = tfidf_strings_train
    test_data['TF-IDF'] = tfidf_strings_test

    # Create a new DataFrame for the feature names
    feature_names_df = pd.DataFrame({feature_name_column: feature_names_list})


    return train_data, test_data
"""


def main():
    
  
    train_file='D:/Projects/Customer-Churn-Prediction/datasets/train_processed.csv'
    

    train_set_file=getfile()
    
    processed_train=batch_processing(pd.read_csv(train_set_file),os.path.splitext(os.path.basename(train_set_file))[0])
    
   
    
    processed_train.to_csv(train_file, index=False)
   


   
if __name__ == "__main__":
    main()

