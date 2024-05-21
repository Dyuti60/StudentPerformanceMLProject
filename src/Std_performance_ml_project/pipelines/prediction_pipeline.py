import sys
from src.Std_performance_ml_project.utils import calculate_Score_Regression
from src.Std_performance_ml_project.exception import MLException
class PredictionPipeline:
    def initiate_prediction_pipeline(self,X_test,y_test,model):
        try:
            y_test_pred=model.predict(X_test)
            mae_test,mse_test,rmse_test,r2_test=calculate_Score_Regression(y_test, y_test_pred)
            return mae_test,mse_test,rmse_test,r2_test
        except Exception as e:
            raise MLException(e,sys)