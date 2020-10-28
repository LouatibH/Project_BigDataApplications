#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Wine Quality Sample
def train():
    import os
    import warnings
    import sys

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from xgboost import XGBClassifier






    import mlflow
    import mlflow.sklearn
    
    import logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    """def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2"""


    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = pd.read_csv('train.csv')


    
    
    """try:
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)"""

    # The predicted column is "quality" which is a scalar from [3, 9]
    #train_x = train.drop(["quality"], axis=1)
    X = csv_url.drop(['TARGET'],axis=1)
    y = csv_url['TARGET']
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
    
    


    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        modelXGB = XGBClassifier()
        model = modelXGB.fit(X_train, y_train)
        
        
    
    
        # evaluate model
        
        y_pred_XGB = modelXGB.predict(X_test)
        predictions_XGB = [round(value) for value in y_pred_XGB]
        
        loss = log_loss(y_test, y_pred_XGB)
        acc = accuracy_score(y_test, predictions_XGB)
             

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})
        
        
        
        

        """# Evaluate Metrics
        predicted_qualities = modelXGB.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
"""
        # Print out metrics
        print("XGBoost model ")
        print("  Loss: %s" % loss)
        print("  Accuracy: %s" % acc)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(modelXGB, "model")
train()


# In[25]:


#get_ipython().system('jupyter nbconvert --to script train.ipynb')


# In[ ]:




