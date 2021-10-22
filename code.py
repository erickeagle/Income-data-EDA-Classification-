#importing libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import sys
import logging
import warnings
warnings.filterwarnings("ignore")




def final_model(preprocessor,final_classifier,X_train, y_train,X_test,y_test):
    '''This function is used for saving the best model and Test dat with actual and predicted values
    '''
    print('IN FINAL STAGE')
    final = Pipeline(steps=[('preprocessor', preprocessor),('classifier',final_classifier['classifier'])])
    final.fit(X_train, y_train)
    filename = 'final_model.pkl'
    pickle.dump(final, open(filename, 'wb'))
    Test=X_test.copy()
    Test['actual_income']=y_test
    y_pred = final.predict(X_test)
    Test['predicted_income']=y_pred
    Test.to_csv('Final_csv.csv')

    
    
def model_building(preprocessor,X_train,y_train):
    '''This function is used for finding the best model and its parameters using hyparparameter tunning
    '''
    print('MODEL BUILDING')
    classifier_pipe = Pipeline(steps=(('preprocessor', preprocessor),["classifier",DecisionTreeClassifier(random_state=11)]))
    classifier_param_grid = [  
                         {
                          "classifier":[DecisionTreeClassifier(random_state=11)],
                          "classifier__criterion":["gini","entropy"],
                          "classifier__max_depth":np.arange(10,21,4),
                          "classifier__min_samples_split":np.arange(2,21,4),
                          "classifier__min_samples_leaf":np.arange(1,10,3)
                         },
                         {
                          "classifier":[RandomForestClassifier(random_state=11,n_jobs=-1,class_weight={' <=50K': 1, ' >50K': 1})],
                          "classifier__criterion":["gini","entropy"],
                          "classifier__n_estimators":np.arange(50,100,10),
                          "classifier__min_samples_split":np.arange(2,16,4),
                          "classifier__min_samples_leaf":np.arange(1,10,3)
                         },   
                         {
                          "classifier":[XGBClassifier(random_state=42,n_jobs=-1)],
                          "classifier__min_child_weight":np.arange(2,16,4),
                          "classifier__gamma":np.arange(1,10,3),
                        "classifier__max_depth":np.arange(10,21,4),
                         },
                        ]
    
    grid_cv = GridSearchCV(estimator=classifier_pipe,param_grid=classifier_param_grid,scoring="accuracy",cv=5)
    grid_cv.fit(X_train,y_train)
    final_classifier = grid_cv.best_estimator_
    return final_classifier



def preprocessing(df):
    '''This function is used for preprocessing the data to prepare the data according to our need
    '''
    print('PREPROCESSING')
    df.drop_duplicates(inplace=True,ignore_index=True)
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('-', '_')

    df.replace({' ?':None},inplace=True)
    df["workclass"].fillna('ffill',inplace=True)
    df["occupation"].fillna('ffill',inplace=True)
    df["native_country"].fillna('ffill',inplace=True)
    X = df.drop(['income','capital_gain', 'capital_loss'],axis=1)
    y = df["income"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=123)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    categorical_features = df.select_dtypes(include=['object']).drop(['income'], axis=1).columns
    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)])
    return X_train, X_test, y_train, y_test,preprocessor



def main(df): 
    '''This is the main function of the program
    '''
    print('IN MAIN')
    X_train, X_test, y_train, y_test,preprocessor=preprocessing(df)
    final_classifier=model_building(preprocessor,X_train,y_train)
    final_model(preprocessor,final_classifier,X_train, y_train,X_test,y_test)



if __name__ == '__main__':
    LOG_FORMAT = '%(levelname)s %(asctime)s - %(message)s'
    logging.basicConfig(filename = 'logs.log',
                        level = logging.DEBUG,
                        format = LOG_FORMAT,
                        filemode = 'w')  
    path=sys.argv
    logging.info('Started fetching')
    main(pd.read_csv(path[1]))
    logging.info('Fetching Done')
         