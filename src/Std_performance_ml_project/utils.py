import os
import sys
from src.Std_performance_ml_project.exception import CustomException
from src.Std_performance_ml_project.logger import logging

import pandas as pd
import numpy as np
#from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,accuracy_score,precision_score,roc_auc_score
from sklearn.model_selection import train_test_split

import pymysql
import pickle

# Set up connection to My SQL : Constant
#load_dotenv()
host = os.getenv('host')
user = os.getenv('user')
password=os.getenv('password')
db=os.getenv('db')

def read_sql_Data():
    logging.INFO('Reading the SQL Database : Started')
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info('Connection Established to Database')
        df=pd.read_sql_query('Select * from Students',mydb)
        return df
    except Exception as e:
        raise CustomException(e)
    
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e)
    
def calculate_Score_Regression(actual, predicted):
    acc=accuracy_score(actual,predicted)
    prec=precision_score(actual,predicted)
    r2=r2_score(actual, predicted)
    roc=roc_auc_score(actual,predicted)
    return acc,prec,r2,roc

def trainTestSplit(X_result,Y_result):
    X_train,X_test,y_train,y_test=train_test_split(X_result,Y_result,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test


def evaluate_models(X_result,Y_result,models,params):
    try:
        report=[]
        models_list=[]
        accuracy_list=[]
        for i in range(len(models)):
            #Have Model Function:
            model=list(models.values())[i]

            #Have parameters for each model:
            param=params[list(models.keys())[i]]

            #train-test split X_result and Y_result:
            X_train,X_test,y_train,y_test=trainTestSplit(X_result,Y_result)

            #Hyperparameter Tuning:
            gs=GridSearchCV(model,param,cv=5)

            #Train Data '.fit':
            gs.fit(X_train,y_train)

            #Predict Data '.predict':
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            #Train Data Model Performance:
            acc_train,prec_train,r2_train,roc_train=calculate_Score_Regression(y_train, y_train_pred)

            #Test Data Model Performance
            acc_test,prec_test,r2_test,roc_test=calculate_Score_Regression(y_test, y_test_pred)

            models_list.append(list(models.keys())[i])
            accuracy_list.append(r2_test)
        report=pd.DataFrame(zip(models_list,accuracy_list),columns=['Model','Accuracy'])
        return report
    except Exception as e:
        raise CustomException(e)

    

