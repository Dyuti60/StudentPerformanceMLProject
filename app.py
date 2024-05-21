from src.Std_performance_ml_project.components.data_ingestion import DataIngestion
from src.Std_performance_ml_project.components.data_transformation import DataTransformation
#from src.Std_performance_ml_project.components.model_trainer import ModelTrainer
#from src.Std_performance_ml_project.components.model_monitoring import ModelMonitoring
from src.Std_performance_ml_project.exception import MLException
from src.Std_performance_ml_project.logger import logging
import numpy as np

import sys


if __name__=="__main__":
    logging.info("The execution has started")

    try:
        DataIngestion=DataIngestion()
        train_data_path,test_data_path,raw_data_path=DataIngestion.initiate_data_ingestion()

        DataTransformation=DataTransformation()
        train_array,test_array,preprocessor_obj=DataTransformation.initiate_DataTransformation(raw_data_path,train_data_path,test_data_path)

        #ModelTrainer=ModelTrainer()
        #ModelTrainer.initiate_model_training(train_array,test_array)

        #ModelMonitoring=ModelMonitoring()
        #ModelMonitoring.initiate_model_training()

    except Exception as e:
        raise  MLException(e,sys)