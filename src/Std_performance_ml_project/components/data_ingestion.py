import os
import sys
from sklearn.model_selection import train_test_split
from src.Std_performance_ml_project.exception import CustomException
from src.Std_performance_ml_project.logger import logging
from src.Std_performance_ml_project.utils import read_sql_Data
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """
    DataIngestionConfig class is used to store the data ingestion configuration --> artifacts of data ingestion process to br stored
    """
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','train.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            df=read_sql_Data()
            logging.info('Fetched records from SQL database')

            # create directories to store the raw dataset,train dataset and test dataset
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)

            #save the dataset in csv format in the directory created above
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            #train test data split
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #Save train and test dataset in train and test directories
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)


            logging.info('Data Ingestion Completed')

            return(self.ingestion_config.raw_data_path,
                   self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e)        
