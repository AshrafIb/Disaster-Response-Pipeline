# Disaster-Response-Pipeline
This Repository is for my Disaster Response Pipeline
# 1. Motivation and Overview
In this project I investigate data from Figure Eight and establish a machine learning algorithm that can classify text data. 
This algorithm predicts disaster messages that are entered into a web application and assigns them to the respective message categories. 
This project consists of two Jupyter notebooks that represent the analysis process, a web app and several py's that build the backend of this app. 
# 2. App Components 
## 2.1 ETL-Process 
Under *data/process_data.py* you will find the steps of the ETL Process, which consists of loading, cleaning and storing the data into a Database. 

## 2.2 ML-Pipeline 

Under *models/train_classifier.py* the data is loaded, processed for ML and trained with an optimized Model (optimization steps are shown in the corresponding jupyter notebook.). 
The following Model achieves the overall Results - columnbased values are shown in the corresponding Notebook:
  + The overall Accuracy score is 93.2%
  + The overall Precision score is 91.5%
  + The overall Recall score is 93.2%
  + The overall F1 score is 90.9%

This model is stored as pickle and used in the WebApp.

## 2.3 WebApp 

To run the WebApp follow the commands in *data/* 

The first Screenshot shows the Main-Page of the App with two distributional plots. 


The Second screenshot shows the results for new Text. 

## 2.4 Requierements

+ Scikit-learn  
+ Numpy 
+ Pandas 
+ Flask 
+ sqlalchemy
+ nltk 
+ seaborn 
+ matplotlib 
+ Plotly 


