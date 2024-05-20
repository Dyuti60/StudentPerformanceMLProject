import os
import sys
from src.Std_performance_ml_project.exception import CustomException
from src.Std_performance_ml_project.logger import logging
from src.Std_performance_ml_project.utils import read_sql_Data
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """
    DataIngestionConfig class is used to store the data ingestion configuration --> artifacts of data ingestion process to br stored
    """
    raw_data_path:str=os.path.join('artifacts','train.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            df=read_sql_Data()
            logging.info('Fetched records from SQL database')

            # create directory to store the raw dataset
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            #save the dataset in csv format in the directory created above
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Data Ingestion Completed')

            return(self.ingestion_config.raw_data_path)
        
        except Exception as e:
            raise CustomException(e)        
