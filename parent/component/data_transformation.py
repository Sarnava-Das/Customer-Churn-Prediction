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
    for dirname, _, filenames in os.walk(source_file.ROOT_DIR): 
        for filename in filenames:
            if(pathlib.Path(os.path.join(dirname, filename)).suffix =='.csv'):
                path.append(os.path.join(dirname, filename))
   
   
    train_set_filename=""
    for filename in path:
        if(os.path.basename(filename)==source_file.TRAIN_SET): 
            train_set_filename=filename
        
    return train_set_filename

def batch_processing(data,filename):
  
    batch_size = 1000  
    processed_data=pd.DataFrame()
    
    if filename==source_file.TRAIN_SET.split('.')[0]:
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
        
            # Get the current batch of data
            batch_data = data.iloc[batch_start:batch_end]
            processed_data = pd.concat([processed_data, process_data(batch_data)])

    return processed_data
 
def main():
  
    train_file=source_file.TRAIN_SET_PROCESSED_PATH
    
    train_set_file=getfile()
    processed_train=batch_processing(pd.read_csv(train_set_file),os.path.splitext(os.path.basename(train_set_file))[0])
    processed_train.to_csv(train_file, index=False)
  
if __name__ == "__main__":
    main()

