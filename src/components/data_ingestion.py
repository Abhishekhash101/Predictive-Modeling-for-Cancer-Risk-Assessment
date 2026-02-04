from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
import os
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path: str = os.path.join("artifacts", "raw.csv")
        self.train_data_path: str = os.path.join("artifacts", "train.csv")
        self.test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Working with data ingestion")
            df=pd.read_csv(r"notebook/data/cancer-risk-factors.csv")
            logging.info("dataset read sucessfully")
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Raw DataSaved Sucessfully")

            train_df,test_df=train_test_split(df,test_size=0.3,random_state=42)

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True
            )
            train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("train_data_path Sucessfully")

            os.makedirs(
                os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True
            )
            test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("test_data_path Sucessfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

    
    
if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()