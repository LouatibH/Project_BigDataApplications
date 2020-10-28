#!/usr/bin/env python
# coding: utf-8

# In[1]:


def train():
    import os
    import warnings
    import sys

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestRegressor







    import mlflow
    import mlflow.sklearn
    
    import logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = pd.read_csv('train.csv')



    # The predicted column is "quality" which is a scalar from [3, 9]
    #train_x = train.drop(["quality"], axis=1)
    X = csv_url.drop(['TARGET'],axis=1)
    y = csv_url['TARGET']
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
    
    


    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        rfmodel = RandomForestRegressor()
        model = rfmodel.fit(X_train, y_train)
        
        
        # evaluate model
        
        y_pred_rf = rfmodel.predict(X_test)
        predictions_rf = [round(value) for value in y_pred_rf]
        
        acc = accuracy_score(y_test, predictions_rf)
             

        # log metrics
        mlflow.log_metrics({"accuracy": acc})
        
        
        
        # Print out metrics
        print("RandomForest model ")
        print("  Accuracy: %s" % acc)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(rfmodel, "model")
train()


# In[ ]:


#get_ipython().system('jupyter nbconvert --to script train_.ipynb')

