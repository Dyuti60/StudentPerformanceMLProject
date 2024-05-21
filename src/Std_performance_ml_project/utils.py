import os
import sys
from src.Std_performance_ml_project.exception import MLException
from src.Std_performance_ml_project.logger import logging

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

#Models:
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import pymysql
import pickle

# Set up connection to My SQL : Constant
load_dotenv()
host = os.getenv('host')
user = os.getenv('user')
password=os.getenv('password')
db=os.getenv('db')

def read_sql_Data():
    logging.info('Reading the SQL Database : Started')
    try:
        mydb=pymysql.connect(
            host='127.0.0.1',
            port=3306,
            user='root',
            password='Satsang@123',
            db='student_details',
        )
        logging.info('Connection Established to Database')
        cursor=mydb.cursor()
        cursor.execute('Select * from student_performance')
        field_names=[i[0] for i in cursor.description]
        df=pd.DataFrame(data=cursor.fetchall(),index=None,columns=field_names)
        mydb.close()
        return df
    except Exception as e:
        raise MLException(e,sys)
    
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise MLException(e,sys)
    
def models_dict():
    models={
        'Linear Regression':LinearRegression(),
        'Lasso':Lasso(),
        'Ridge':Ridge(),
        'SVR':SVR(),
        'KNN':KNeighborsRegressor(),
        'Decision Tree Regression':DecisionTreeRegressor(),
        'Random Forest Regression':RandomForestRegressor(),
        'Gradient Boosting Regression':GradientBoostingRegressor(),
        'Ada Boost Regression':AdaBoostRegressor(),
        'XGBoost Regression':XGBRegressor(),
        'CatBoot Regression':CatBoostRegressor()
    }
    return models

def hyperparameterTuning_Params():
    params={
        'Linear Regression':{},
        'Lasso':{'alpha':[1.0,0.2,0.4,0.6,2.0,4.0,5.0]
                 },
        'Ridge':{
            'alpha':[1.0,0.2,0.4,0.6,2.0,4.0,5.0]
        },
        'SVR':{
            'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'C':[1,2,3,4,5,6,7,8,9,10],
            'epsilon':[0.1,0.2,0.5,0.8,1.0,1.2] 
        },
        'KNN':{
            'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        },
        'Decision Tree Regression':{
            'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter':['best', 'random'],
            'ccp_alpha':[1,2,3,4,5,6],
            'max_features':['sqrt','log2']
        },
        'Random Forest Regression':{
            'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'n-estimators':[8,16,32,64,128],
            'max_features':['sqrt','log2']
        },
        'Gradient Boosting Regression':{
            # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate':[.1,.01,.05,.001],
            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            # 'criterion':['squared_error', 'friedman_mse'],
            # 'max_features':['auto','sqrt','log2'],
            'n_estimators': [8,16,32,64,128,256]
        },
        'XGBoost Regression':{
            'learning_rate':[.1,.01,.05,.001],
            'n_estimators': [8,16,32,64,128,256]
        },

        "CatBoosting Regressor":{
            'depth': [6,8,10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100]
        },
        "AdaBoost Regressor":{
            'learning_rate':[.1,.01,0.5,.001],
            # 'loss':['linear','square','exponential'],
            'n_estimators': [8,16,32,64,128,256]
        }
    }
    return params

def calculate_Score_Regression(actual, predicted):
    mae=mean_absolute_error(actual,predicted)
    mse=mean_squared_error(actual,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(actual, predicted)
    
    return mae,mse,rmse,r2

def trainTestSplit(X_result,Y_result):
    X_train,X_test,y_train,y_test=train_test_split(X_result,Y_result,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test


def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try:
        report=[]
        models_list=[]
        accuracy_list=[]
        for i in range(len(models)):
            #Have Model Function:
            model=list(models.values())[i]

            #Have parameters for each model:
            param=params[list(models.keys())[i]]


            #Hyperparameter Tuning:
            gs=GridSearchCV(model,param,cv=5)

            #Train Data '.fit':
            gs.fit(X_train,y_train)

            #Predict Data '.predict':
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            #Train Data Model Performance:
            mae_train,mse_train,rmse_train,r2_train=calculate_Score_Regression(y_train, y_train_pred)

            #Test Data Model Performance
            mae_test,mse_test,rmse_test,r2_test=calculate_Score_Regression(y_test, y_test_pred)

            models_list.append(list(models.keys())[i])
            accuracy_list.append(r2_test)
        report=pd.DataFrame(zip(models_list,accuracy_list),columns=['Model','Accuracy'])
        return report
    except Exception as e:
        raise MLException(e,sys)

    

