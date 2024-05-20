import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Null Value Handle
from sklearn.impute import SimpleImputer
# Scaling  
from sklearn.preprocessing import StandardScaler,OneHotEncoder
# Pipleine
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Train Test Split:


from src.Std_performance_ml_project.utils import save_object
from src.Std_performance_ml_project.exception import CustomException
from src.Std_performance_ml_project.logger import logging

@dataclass
class DataTransformationConfig:
    '''
    This class is used to store the configuration of data transformation --> directory name to store data transformation artifacts. 
    '''
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self,raw_data_path):
        '''
        This function will return the data transformation object
        Handle missing data -> SimpleImputer
        Handle Categorical Data -> OneHotEncoder
        Scale data (numerical and categorical) --> StandardScaler
        '''
        try:
            #fetch data:
            raw_data=pd.read_csv(raw_data_path)

            # Divide into Numerical and Categorical features:
            numerical_features=raw_data.select_dtypes(include=['int64','float64']).columns
            categorical_features=raw_data.select_dtypes(include=['object']).columns

            #Numerical feature Pipeline
            num_pipeline=Pipeline(steps=[
                ('simpleImpute',SimpleImputer(strategy='median')),
                ('Scaling',StandardScaler())
            ])
            #Categorical feature Pipeline
            cat_pipeline=Pipeline(steps=[
                ('simpleImpute',SimpleImputer(strategy='most_frequent')),
                ('Encoding',OneHotEncoder()),
                ('Scaling',StandardScaler())
            ])
            #Column Transformer
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e)

    def initiate_DataTransformation(self,raw_data_path,train_data_path,test_data_path):
        '''
        This function will initiate the data transformation process
        '''
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            logging.info('Reading data from test and train file path')

            #Get the data transformation object:
            preprocessor_obj=self.get_data_transformation_object(self,raw_data_path)

            #Target Feature:
            target_feature='math_score'

            #Divide the train dataset:
            input_feature_train_df=train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df=train_df[target_feature]

            #Divide the test dataset:
            input_feature_test_df=test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df=test_df[target_feature]

            logging.info("Applying preprocessing into train and test datasets - Input features only")
            
            # Applying preprocessing to input features of train and test datasets
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            #Combine the input and target features into train array
            train_arr=np.concatenate(input_feature_train_arr,np.array(target_feature_train_df))
            
            #Combine the input and target features into test array
            test_arr=np.concatenate(input_feature_test_arr,np.array(target_feature_test_df))

            save_object(filepath=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)

            return (
                train_arr,
                test_arr,
                preprocessor_obj
            )

        except Exception as e:
            raise CustomException(e)