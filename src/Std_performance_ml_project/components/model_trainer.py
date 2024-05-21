import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

#Regression Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


from src.Std_performance_ml_project.exception import MLException
from src.Std_performance_ml_project.logger import logging
from src.Std_performance_ml_project.utils import save_object,evaluate_models,calculate_Score_Regression, models_dict,hyperparameterTuning_Params

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifacts','model.pkl')
    trainer_model_report_file_path=os.path.join('artifacts','model_report.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            X_train, X_test, y_train, y_test = (train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1])
            models=models_dict()
            params=hyperparameterTuning_Params()
            model_report=evaluate_models(X_train,X_test,y_train,y_test,models,params)
            
            ## To get best model score from dataframe
            best_model_score = str(model_report['Score'][0])

             ## To get best model name from dataframe
            best_model_name = str(model_report['Model'][0])

            ## To get the best model function from dataframe
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)

            ## To get the hyperparameter for the best model
            best_params = params[best_model_name]
            
            
            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )
            model_report.to_csv(self.model_trainer_config.trainer_model_report_file_path)

            return(
                X_test,
                y_test,
                best_model_name,
                best_model,
                best_model_score,
                best_params
            )
        except Exception as e:
            raise MLException(e,sys)