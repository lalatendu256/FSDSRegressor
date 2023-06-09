# In Data Ingestion components we need to do how to read the data

import os # to create the file path.
import sys # system error
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_tranformation import DataTransformation

## Initialize Data Ingestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## Create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initate_data_ingestion(self):
        logging.info('Data Ingestion methods starts.')
        try:
            df=pd.read_csv('./notebooks/data/gemstone.csv')

            logging.info(os.path.join('Dataset read as Pandas Dataframe'))
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            logging.info('Saving the dataset in raw file')
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion is completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage.')
            raise CustomException(e,sys)   

    

