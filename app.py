from src.Std_performance_ml_project.components.data_ingestion import DataIngestion
from src.Std_performance_ml_project.components.data_transformation import DataTransformation
from src.Std_performance_ml_project.components.model_trainer import ModelTrainer
from src.Std_performance_ml_project.pipelines.training_pipeline import TrainingPipeline
from src.Std_performance_ml_project.pipelines.prediction_pipeline import PredictionPipeline
from src.Std_performance_ml_project.components.model_monitoring import ModelMonitoring
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

        ModelTrainer=ModelTrainer()
        X_train,X_test,y_train,y_test,best_model_name,best_model,best_model_score,best_params=ModelTrainer.initiate_model_training(train_array,test_array)

        TrainingPipeline=TrainingPipeline()
        GS=TrainingPipeline.initiate_training_Pipeline(X_train,y_train,best_model,best_params)

        PredictionPipeline=PredictionPipeline()
        mae_test,mse_test,rmse_test,r2_test=PredictionPipeline.initiate_prediction_pipeline(X_test,y_test,GS)

        ModelMonitoring=ModelMonitoring()
        ModelMonitoring.initiate_model_training(rmse_test,mse_test,r2_test,mae_test,best_model_name,best_model,best_model_score,best_params)

    except Exception as e:
        raise  MLException(e,sys)