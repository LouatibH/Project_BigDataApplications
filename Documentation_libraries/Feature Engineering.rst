.. code:: ipython3

    import pandas as pd
    import numpy as np
    from numpy import loadtxt

.. code:: ipython3

    application_train = pd.read_csv('application_clean_train.csv')
    application_test = pd.read_csv('application_test.csv')

Average on null value for quantitive variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    quantitative = application_train.select_dtypes(include=np.number).apply(lambda x: x.fillna(x.mean()) ,axis=0)
    qualitative = application_train.select_dtypes(exclude=np.number)
    application_train = qualitative.join(quantitative)

Encoding
~~~~~~~~

.. code:: ipython3

    application_train = pd.get_dummies(application_train)
    application_test = pd.get_dummies(application_test)
    
    print('Shape of the train  ', application_train.shape)
    print('Shape of the test ', application_test.shape)


.. parsed-literal::

    Shape of the train   (307511, 206)
    Shape of the test  (48744, 242)


Keep same variable in train and test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    label = application_train['TARGET']
    application_train, application_test = application_train.align(application_test, join='inner', axis=1)
    application_train['TARGET'] = label
    
    print('Shape of the train :', application_train.shape)
    print('Shape of the train :', application_test.shape)


.. parsed-literal::

    Shape of the train : (307511, 202)
    Shape of the train : (48744, 201)


Correlation
~~~~~~~~~~~

.. code:: ipython3

    correlation = application_train.corr()['TARGET'].sort_values()
    print('Prositive top 20 :\n',correlation.tail(20))
    print('\n\nNegative top 20 :\n',correlation.head(20))


.. parsed-literal::

    Prositive top 20 :
     NAME_HOUSING_TYPE_With parents                       0.029966
    NAME_CONTRACT_TYPE_Cash loans                        0.030896
    DEF_60_CNT_SOCIAL_CIRCLE                             0.031251
    DEF_30_CNT_SOCIAL_CIRCLE                             0.032222
    LIVE_CITY_NOT_WORK_CITY                              0.032518
    DAYS_REGISTRATION                                    0.041975
    FLAG_DOCUMENT_3                                      0.044346
    REG_CITY_NOT_LIVE_CITY                               0.044395
    FLAG_EMP_PHONE                                       0.045982
    NAME_EDUCATION_TYPE_Secondary / secondary special    0.049824
    REG_CITY_NOT_WORK_CITY                               0.050994
    DAYS_ID_PUBLISH                                      0.051457
    CODE_GENDER_M                                        0.054713
    DAYS_LAST_PHONE_CHANGE                               0.055218
    NAME_INCOME_TYPE_Working                             0.057481
    REGION_RATING_CLIENT                                 0.058899
    REGION_RATING_CLIENT_W_CITY                          0.060893
    DAYS_EMPLOYED                                        0.070075
    DAYS_BIRTH                                           0.078239
    TARGET                                               1.000000
    Name: TARGET, dtype: float64
    
    
    Negative top 20 :
     EXT_SOURCE_2                           -0.160303
    EXT_SOURCE_3                           -0.157397
    EXT_SOURCE_1                           -0.099152
    NAME_EDUCATION_TYPE_Higher education   -0.056593
    CODE_GENDER_F                          -0.054704
    NAME_INCOME_TYPE_Pensioner             -0.046209
    ORGANIZATION_TYPE_XNA                  -0.045987
    AMT_GOODS_PRICE                        -0.039628
    REGION_POPULATION_RELATIVE             -0.037227
    NAME_CONTRACT_TYPE_Revolving loans     -0.030896
    AMT_CREDIT                             -0.030369
    FLOORSMAX_AVG                          -0.029145
    FLOORSMAX_MEDI                         -0.028989
    FLOORSMAX_MODE                         -0.028631
    FLAG_DOCUMENT_6                        -0.028602
    NAME_HOUSING_TYPE_House / apartment    -0.028555
    NAME_FAMILY_STATUS_Married             -0.025043
    HOUR_APPR_PROCESS_START                -0.024166
    FLAG_PHONE                             -0.023806
    NAME_INCOME_TYPE_State servant         -0.023447
    Name: TARGET, dtype: float64


.. code:: ipython3

    application_train.to_csv('train.csv')
    application_test.to_csv('test.csv')
