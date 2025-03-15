import os
import pandas as pd 
from sklearn.model_selection import train_test_split
import logging

#make a directory to store logs
logs_dir = 'logs'
os.makedirs(logs_dir,exist_ok=True)

#setting up logger for both console and file
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(logs_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)

#Finished setting up logger and now data load function

def load_data(url:str) -> pd.DataFrame:
    """ Loads the CSV file from given URL. """
    try:
        df = pd.read_csv(url)
        logger.debug('Data loaded from %s',url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse CSV file %s',e)
        raise
    except Exception as e:    
        logger.error('Unknown Error while loading data %s',e)
        raise

def preprocess_data(df : pd.DataFrame) -> pd.DataFrame :
    """ dropping unnecessary columns and remaning columns as text and target"""
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data is preprocessed')
        return df 
    except KeyError as e:
        logger.error("Missing column in the Dataframe %s",e)
        raise
    except Exception as e:
        logger.error('Unknown error while preprocessing dataframe%s',e)
        raise

def save_data(train_data : pd.DataFrame , test_data : pd.DataFrame , data_path : str) ->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'),index=False)    
        test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'),index=False)   
        logger.debug('Train and test data saved to raw folder in %s',data_path)
    except Exception as e:
        logger.error('Unexpected Error while saving Data %s',e)
        raise

def main():
    try:
        test_size = 0.2
        data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df=load_data(url = data_url)
        preprocessed_df = preprocess_data(df)
        train_data,test_data = train_test_split(preprocessed_df,test_size=test_size,random_state=2)
        save_data(train_data=train_data,test_data=test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion%s',e)
        raise

if __name__ == '__main__':
    main()