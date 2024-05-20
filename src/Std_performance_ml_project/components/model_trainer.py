import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

#Regression Metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


from src.Std_performance_ml_project.exception import CustomException
from src.Std_performance_ml_project.logger import logging
from src.Std_performance_ml_project.utils import save_object,evaluate_models,calculate_Score_Regression, models_dict,hyperparameterTuning_Params

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            X_train, X_test, y_train, y_test = (train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1])
            models=models_dict()
            params=hyperparameterTuning_Params()
            model_report:dict=evaluate_models(X_train,X_test,y_train,y_test,models,params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print("This is the best model:")
            print(best_model_name)

            model_names = list(params.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]
            
            return(
                best_model,
                best_model_score,
                model_report,
                best_params
            )
        except Exception as e:
            raise CustomException(e,sys)