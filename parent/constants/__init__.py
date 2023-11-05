import os

ROOT_DIR = os.getcwd()  #to get current working directory

# Data ingestion related variables
DATASET_IDENTIFIER = 'shantanudhakadd/bank-customer-churn-prediction'
DATASET_DIR = "datasets"
DATASET_DESTINATION_PATH = os.path.join(ROOT_DIR,DATASET_DIR)

# Data transformation,evaluation related variables
COLUMN1_IMPUTE = 'RowNumber'
COLUMN2_IMPUTE='CustomerId'
COLUMN3_IMPUTE ='Surname'
COLUMN1_ENCODE='Geography'
COLUMN2_ENCODE='Gender'

TRAIN_SET='Churn_Modelling.csv'
TRAIN_SET_PROCESSED_NAME='train_processed.csv'
TRAIN_SET_PROCESSED_PATH=os.path.join(ROOT_DIR,DATASET_DIR,TRAIN_SET_PROCESSED_NAME)



# Model pusher related variables
MODEL_DIR='models'
PRED_MODEL_NAME='model.pkl'
TFIDF_NAME='tfidf_vectorizer.pkl'
ENCODING_NAME='label_encoding.pkl'
ENCODING_PATH=os.path.join(ROOT_DIR,MODEL_DIR,ENCODING_NAME)
TFIDF_PATH=os.path.join(ROOT_DIR,MODEL_DIR,TFIDF_NAME)
PRED_MODEL_PATH=os.path.join(ROOT_DIR,MODEL_DIR,PRED_MODEL_NAME)

