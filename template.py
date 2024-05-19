import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
project_name='Std_performance_ml_project'

list_of_files=[
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/components/data_ingestion.py',
    f'src/{project_name}/components/data_transformation.py',
    f'src/{project_name}/components/model_trainer.py',
    f'src/{project_name}/components/model_monitoring.py',
    f'src/{project_name}/pipelines/__init__.py',
    f'src/{project_name}/pipelines/training_pipeline.py',
    f'src/{project_name}/pipelines/prediction_pipeline.py',
    f'src/{project_name}/exception.py',
    f'src/{project_name}/logger.py',
    f'src/{project_name}/utils.py',
    'main.py',
    'app.py',
    'Dockerfile.py',
    'requirements.txt',
    'setup.py'
]

for file in list_of_files:
    filepath=Path(file)
    filedir,filename=os.path.split(filepath)
    print(filepath)

    #creates only directory if exists or not
    if filedir !='':
        print(filedir)
        print(filename)
        os.makedirs(filedir,exist_ok=True)
        logging.info(f'Directory {filedir} for file - {filename} created')

    #checks if filepath doesn't exists or if filepath size is zero then, create empty file else does nothing
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as file_obj:
            pass
            logging.info(f'creating empty file :{filepath}')
    else:
        logging.info(f'File {filename} already exists')