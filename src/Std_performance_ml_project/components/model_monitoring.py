import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from src.Std_performance_ml_project.exception import MLException
from src.Std_performance_ml_project.logger import logging
from src.Std_performance_ml_project.utils import calculate_Score_Regression,save_object

# mlflow
class ModelMonitoring:
    def initiate_model_training(self,X_test,y_test,best_model_name,best_model,best_model_score,best_params):
        try:
            mlflow.set_registry_uri("https://dagshub.com/krishnaik06/mlprojecthindi.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (mae,mse,rmse,r2) = calculate_Score_Regression(y_test, predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score<0.6:
                raise MLException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")



        except Exception as e:
            raise MLException(e,sys)