import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.Std_performance_ml_project.exception import MLException
class TrainingPipeline:
    def initiate_training_Pipeline(self,X_train,y_train,best_model,best_params):
        try:
            GS=GridSearchCV(best_model,best_params,cv=5)
            GS.fit(X_train,y_train)

            return GS

        except Exception as e:
            raise MLException(e,sys)