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
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Evaluate:
    
    def __init__(self,models,model_type = 'ranf'):
        self.models_eval =[]
        self.model_type = model_type
        
    def evaluate(self,X_test,y_test,metrics='r2_score'):
        
        try:
            X = X_test.drop('Video Name',1)
        except:
            X = X_test
        for i in range(y_test.shape[1]):
            print(y_test.columns[i])
            preds = self.models_eval[i].predict(X)
            if metrics == 'r2_score':
                r2score = r2_score(y_test.iloc[:,i],preds)
                print("R2_SCORE %f" % r2score,"\n")
            elif metrics == 'rmse':
                rmse = math.sqrt(mean_squared_error(preds,y_test.iloc[:,i]))
                print("RMSE %f" % rmse,"\n")
                
    def plot_feature_importance(self,models,figsize,output_columns,feature_columns,savefig=False):
        self.models_eval = models
        try: 
            feature_columns = feature_columns.remove('Video Name')
        except:
            pass
        
        if self.model_type == 'ranf' or self.model_type == 'DT':
            for i in range(len(output_columns)):
                print(output_columns[i])
                importances = self.models_eval[i].feature_importances_
                if self.model_type == 'ranf':
                    std = np.std([tree.feature_importances_ for tree in self.models_eval[i].estimators_],
                                 axis=0)
                indices = np.argsort(importances)[::-1]

                # Print the feature ranking
                print("Feature ranking:")

                for f in range(len(feature_columns)):
                    print("%d. %s (%f)" % (f + 1, feature_columns[indices[f]], importances[indices[f]]))
                    # print("%s" % (feature_columns[indices[f]]))
                    

                # Plot the feature importances of the forest
                
                plt.title(output_columns[i])
                plt.barh(range(len(indices[0:8])), importances[indices[0:8]], color='b', align='center')
                plt.yticks(range(len(indices[0:8])), [feature_columns[i] for i in indices[0:8]])
                plt.xlabel('Relative Importance')
#                 plt.title("Feature importances")
                if savefig:
                    plt.savefig(feature_columns[i]+'_'+ self.model_type+'.png')
                plt.show()
                
        else:
            for i in range(len(output_columns)):
                xgb.plot_importance(self.models_eval[i],max_num_features = 8,xlabel='Gain',title =output_columns[i]+ ' Importance plot')
                importances = self.models_eval[i].feature_importances_
                indices = np.argsort(importances)[::-1]
                for f in range(len(feature_columns)):
                    print("%d. %s (%f)" % (f+1,feature_columns[indices[f]],importances[indices[f]]))
                plt.rcParams['figure.figsize'] = [30,15]
                plt.rcParams['figure.dpi'] = 100
                matplotlib.rcParams.update({'font.size': 22})
                if savefig:
                    plt.savefig(feature_columns[i]+'_XGB_Plot.png')
                plt.show()
        
        