import os # to create the file path.
import sys # system erropr
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer





if __name__=='__main__':    
    obj=DataIngestion() 
    train_data_path,test_data_path=obj.initate_data_ingestion()  
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_tranformation(train_data_path,test_data_path) 
    model_trainer=ModelTrainer() 
    model_trainer.initate_model_training(train_arr,test_arr)