
import pandas as pd
import itertools
import numpy as np
import copy,re


import math
import copy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV,cross_validate,learning_curve,validation_curve
from sklearn.metrics import mean_squared_error, r2_score
import xlrd, os, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib

from copy import deepcopy

from sklearn import tree
import graphviz

import warnings
warnings.filterwarnings('ignore')



from Evaluate import Evaluate
from DataPreprocessing import DataPreprocessing
from Agg_Frame_Data import Agg_Frame_Data

class Model(Evaluate):
    
    # model={"ranf","xgb","DT"}
    def __init__(self,model='ranf',_type="reg"):
        super().__init__(self,model)
        self.preprocessing = DataPreprocessing()
        # self.frameData = Agg_Frame_Data()
        self.model_type = model
        self.type= _type
        self.parameters =[]
        self.all_models=[]
        self.output_length= None
        self.output_columns = None
        self.feature_columns = None
        
        
        if self.model_type == 'ranf':
            self.model = RandomForestRegressor(warm_start=True,verbose=1,random_state=123)
            Unaided_Branding_params = {'bootstrap': False, 'max_depth': 11, 'max_features': 0.25, 'min_samples_leaf': 2, 'n_estimators': 100}
            Brand_mean_cues_params = {'bootstrap': False, 'max_depth': 15, 'max_features': 0.25, 'min_samples_leaf': 2, 'n_estimators': 250}
            Aided_Branding__Mean_params ={'bootstrap': False, 'max_depth': 15, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 100}
            Active_Involvement__Mean_params = {'bootstrap': False, 'max_depth': 14, 'max_features': 0.25, 'min_samples_leaf': 2, 'n_estimators': 300}
            New_Information__Mean_params = {'bootstrap': False, 'max_depth': 13, 'max_features': 0.4, 'min_samples_leaf': 2, 'n_estimators': 200}
            Enjoyment__Mean_params= {'bootstrap': False, 'max_depth': 12, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 300}
            Brand_Appeal__Mean_params = {'bootstrap': False, 'max_depth': 12, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 300}
            Understanding__Mean_params = {'bootstrap': False, 'max_depth': 15, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 150}
            Relevance_of_Information__Mean_params = {'bootstrap': False, 'max_depth': 12, 'max_features': 0.25, 'min_samples_leaf': 2, 'n_estimators': 150}
            Credibility_of_Information__Mean_params = {'bootstrap': False, 'max_depth': 11, 'max_features': 0.25, 'min_samples_leaf': 2, 'n_estimators': 100}
            Brand_Difference__Mean_params= {'bootstrap': False, 'max_depth': 13, 'max_features': 0.25, 'min_samples_leaf': 2, 'n_estimators': 400}
            Interest_peak_params = {'bootstrap': False, 'max_depth': 13, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 200}
            Interest_mean_params = {'bootstrap': False, 'max_depth': 15, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 200}
            Purchase_intent_params ={'bootstrap': False, 'max_depth': 15, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 400}
            Persuasion_mean_params = {'bootstrap': False, 'max_depth': 13, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 200}
            Persuasion_likely_params = {'bootstrap': False, 'max_depth': 20, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 400}
            Interest_frames_params= {'bootstrap': False, 'max_depth': 20, 'max_features': 0.3, 'min_samples_leaf': 2, 'n_estimators': 400}
            
        elif self.model_type == 'xgb':
            self.model = xgb.XGBRegressor(eta=0.3,save_period=1,random_state=123)
            #### RMSE was decreasing with increaasing n_estimators
            Unaided_Branding_params = {'colsample_bytree': 0.4, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 2000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Brand_mean_cues_params = {'colsample_bytree': 0.2, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Aided_Branding__Mean_params ={'colsample_bytree': 0.2, 'max_depth': 15, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Active_Involvement__Mean_params = {'colsample_bytree': 0.6, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 900,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            New_Information__Mean_params = {'colsample_bytree': 0.4, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":10,"learning_rate":0.01,"reg_lambda":1}
            Enjoyment__Mean_params= {'colsample_bytree': 0.8, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Brand_Appeal__Mean_params = {'colsample_bytree': 0.6, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Understanding__Mean_params = {'colsample_bytree': 0.6, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Relevance_of_Information__Mean_params = {'colsample_bytree': 0.2, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Credibility_of_Information__Mean_params = {'colsample_bytree': 0.3, 'max_depth': 10, 'min_samples_leaf': 2, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Brand_Difference__Mean_params= {'colsample_bytree': 0.2, 'max_depth': 8, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Interest_peak_params = {'colsample_bytree': 0.2, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Interest_mean_params = {'colsample_bytree': 0.2, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Purchase_intent_params = {'colsample_bytree': 0.2, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Persuasion_mean_params = {'colsample_bytree': 0.4, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Persuasion_likely_params = {'colsample_bytree': 0.4, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            Interest_frames_params= {'colsample_bytree': 0.4, 'max_depth': 10, 'min_samples_leaf': 4, 'n_estimators': 1000,"objective":'reg:linear',"silent":False,"alpha":1,"learning_rate":0.01,"reg_lambda":1}
            
            
        elif self.model_type == 'DT':
            self.model = DecisionTreeRegressor(random_state=123)
            Unaided_Branding_params = {'splitter':'best','max_depth': 20, 'max_features': 0.6, 'min_samples_leaf': 2,'presort':True}
            Brand_mean_cues_params = {'splitter':'best','max_depth':20, 'max_features': 0.6, 'min_samples_leaf': 2,'presort':True}
            Aided_Branding__Mean_params ={'splitter':'best','max_depth': 20, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Active_Involvement__Mean_params = {'max_depth': 20, 'max_features': 0.3, 'min_samples_leaf': 2,'presort':True}
            New_Information__Mean_params = {'max_depth': 20, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Enjoyment__Mean_params= {'max_depth': 20, 'max_features': 0.3, 'min_samples_leaf': 2,'presort':True}
            Brand_Appeal__Mean_params = {'max_depth': 20, 'max_features': 0.3, 'min_samples_leaf': 2,'presort':True}
            Understanding__Mean_params = {'max_depth': 30, 'max_features': 0.3, 'min_samples_leaf': 2,'presort':True}
            Relevance_of_Information__Mean_params = {'max_depth': 30, 'max_features': 0.3, 'min_samples_leaf': 2,'presort':True}
            Credibility_of_Information__Mean_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Brand_Difference__Mean_params= {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Interest_peak_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Interest_mean_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Purchase_intent_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Persuasion_mean_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Persuasion_likely_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            Interest_frames_params = {'max_depth': 30, 'max_features': 0.2, 'min_samples_leaf': 2,'presort':True}
            
        self.training_params = {"Unaided_Branding":Unaided_Branding_params,"Brand_Cues__Mean":Brand_mean_cues_params,"Aided_Branding__Mean":Aided_Branding__Mean_params,
                        "Active_Involvement__Mean":Active_Involvement__Mean_params,"New_Information__Mean":New_Information__Mean_params,
                        "Enjoyment__Mean":Enjoyment__Mean_params,"Brand_Appeal__Mean":Brand_Appeal__Mean_params,"Understanding__Mean":Understanding__Mean_params,
                        "Relevance_of_Information__Mean":Relevance_of_Information__Mean_params,"Credibility_of_Information__Mean":Credibility_of_Information__Mean_params,
                        "Brand_Difference__Mean":Brand_Difference__Mean_params,"Interest_peak":Interest_peak_params,"Interest_mean_score":Interest_mean_params,
                        "Purchase_intent":Purchase_intent_params,"Persuasion_mean":Persuasion_mean_params,"Persuasion_very_likely":Persuasion_likely_params,"Interest_peak_frames":Interest_frames_params}
            
    def gridSearchCV(self,X_df,y_df,parameters,cv,n_jobs=-1):
        try:
            X, y = X_df.drop('Video Name',1),y_df
        except:
            X, y = X_df,y_df
        print("Search starts")
        hyper_tuning = GridSearchCV(self.model,param_grid=parameters,cv=cv,return_train_score=True,verbose=1,n_jobs=n_jobs,scoring='r2')
        hyper_tuning.fit(X,y)
        print("Search End")
        return hyper_tuning,hyper_tuning.best_params_ 
    
    def plot_tree(self,savefig=False):
        if not isinstance(self.output_length,type(None)):
            for i,target in enumerate(self.output_columns):
                dot_data = tree.export_graphviz(self.all_models[i], out_file=None,
                                                feature_names=self.feature_columns,
                                                class_names=target,
                                                filled=True, rounded=True,special_characters=True)
                graph = graphviz.Source(dot_data)  
                if savefig:
                    graph.render('./plots/'+target)
        else:
            print("Please train your model first\n")
    
    def plot_training_curve(self,estimator,X_train,y_train,X_test,y_test,metric='r2'):
        train_results = estimator.evals_result()
        epochs = len(train_results['validation_0']['rmse'])
        x_axis = range(0, epochs)   
        if metric=='r2':
            TSS_1 = ((y_train-y_train.mean())**2).sum()
            TSS_2 = ((y_test-y_test.mean())**2).sum()

            RSS_1 = (np.array(train_results['validation_0']['rmse'])**2)*X_train.shape[0]
            RSS_2 = (np.array(train_results['validation_1']['rmse'])**2)*X_test.shape[0]
            train_results['validation_0'][metric] = 1- RSS_1/TSS_1
            train_results['validation_1'][metric] = 1- RSS_2/TSS_2
        # plot log loss
        fig, ax = plt.subplots()
        ax.plot(x_axis, train_results['validation_0'][metric], label='Train')
        ax.plot(x_axis, train_results['validation_1'][metric], label='Test')
        prev_score=0
        cnt=0
        for i,score_train in enumerate(train_results['validation_0'][metric]):
            score_test = train_results['validation_1'][metric][i]
            curr_score = abs(score_test-score_train)
            if curr_score >prev_score and curr_score>0.05:
                cnt+=1
            else:
                cnt =0
            if cnt>3:
                epochs= i
                break
            prev_score = curr_score
            
        ax.legend()
        plt.ylim([0,1])
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.ylabel(metric+' Score')
        plt.xlabel('epochs')
        plt.title('XGBoost '+metric+' Score')
        plt.show()
        return epochs
    
    def plot_learning_curve(self,estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1,learningCurve=True):
        plt.figure()
        plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
        plt.ylabel("Score")
        print("Plotting Learning Curve ....\n")
        param_range = np.arange(100,2000,400)
        if learningCurve:
            train_sizes,train_scores, test_scores = learning_curve(estimator, X, y,train_sizes=np.linspace(.1, 1.0, 5),cv=cv, n_jobs=n_jobs)
        else:
            train_scores, test_scores = validation_curve(estimator, X, y, param_name='n_estimators',param_range=param_range,cv=cv, n_jobs=n_jobs,scoring='neg_mean_squared_error')
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        if learningCurve:
            x_var = train_sizes
            label = "Training Sizes"
        else:
            x_var = param_range
            label = "Number of trees"
        plt.grid()
        plt.xlabel(label)
        

        plt.fill_between(x_var, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(x_var, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(x_var, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(x_var, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt
    
    
#     def adaptive learning(self,)
    def train(self,X_df,Y_df,X_test,Y_test,params={},set_params=False,metrics='r2',plot_error=True,retrain=False):
        try:
            Y_df = Y_df.drop('Video Name',1)
            self.feature_columns = X_df.drop('Video Name',1).columns

        except:
            self.feature_columns = X_df.columns        
            pass
        self.all_models=[]
        self.output_length = len(Y_df.columns)
        self.output_columns = Y_df.columns
        prev_model =None
        if set_params:
            self.training_params = params
           
        for i in range(self.output_length):
#             if i>=12:
                if retrain:
                    temp_model = self.all_models[i]
                    prev_model = temp_model
                else:
                    temp_model = copy.deepcopy(self.model)
                print(Y_df.columns[i])
                try:
                    X, y = X_df.drop('Video Name',1),Y_df[Y_df.columns[i]]
                    test_X, test_y = X_test.drop('Video Name',1),Y_test[Y_test.columns[i]]

                except:
                    X,y = X_df,Y_df[Y_df.columns[i]]   
                    test_X, test_y = X_test,Y_test[Y_test.columns[i]]


                if not retrain:
                    temp_model.set_params(**self.training_params[Y_df.columns[i]])
                results= cross_validate(temp_model,X,y,return_estimator=True,verbose=1,cv=5,n_jobs=-1,scoring=metrics) 
    #             print(results.keys())
                index = np.argmax(results['test_score'])
                if self.model_type== 'ranf' or self.model_type =='DT':
                    temp_model.fit(X,y)
                elif self.model_type== 'xgb':
                    if not (isinstance(prev_model,type(None))):
                        prev_model.save_model('model_')
                        temp_model.load_model('model_')
                        prev_model = 'model_'
                    temp_model.fit(X, y,eval_metric=["rmse"], eval_set=[(X,y),(test_X,test_y)],verbose=False,xgb_model=prev_model)
                if retrain:
                    self.all_models[i] = temp_model
                else:
                    self.all_models.append(temp_model)
                    
#                     for i,row in enumerate(self.all_models[-1].predict(test_X)):
#                         print(row,test_y.iloc[i])
    #                 print(temp_model)
                print("For training set")
                print(metrics+"_score: %f" % (np.mean(results['test_score'])))
                print("For test set")
                print(metrics+"_score: %f" % (r2_score(test_y,self.all_models[-1].predict(test_X))))
    #             eval_set = [(X, y)]

                if (self.model_type =='ranf' or self.model_type == 'DT') and plot_error:
                    plT = self.plot_learning_curve(self.all_models[-1],"Training and Testing",X,y,cv=5)
                    plT.show()
                elif self.model_type=='xgb' and plot_error:
                    epochs = self.plot_training_curve(self.all_models[-1],X,y,test_X,test_y)

                print("\n")
        return self.all_models
            
    def predict(self,X_test):
        
        if isinstance(self.output_length,int)==False:
            print("Please train your model first\n")
            return
        else:
            try:
                X = X_test.drop('Video Name',1)
            except:
                X = X_test
            preds= pd.DataFrame()
            for i in range(self.output_length):
                preds[self.output_columns[i]] = self.all_models[i].predict(X)
            
            return preds