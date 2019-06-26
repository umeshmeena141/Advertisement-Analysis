from featureEngineering import featureEngineering

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

class DataPreprocessing(featureEngineering):
    
    def __init__(self):
        super().__init__()
        
    def checkNull(self,df):
        return self.nullPer(df).sort_values(ascending=False)
    
    def removeNULLColumns(self,df,percentage=0.7):
        return df.loc[:, self.nullPer(df)< percentage*100]
    
    def replace(self,df,replace_this,replace_with):
        return df.replace(replace_this,replace_with)
    
    def imputeMedian(self,df,columns=None):
        if not isinstance(columns,list):
            columns = ["Related_Brand"+col for col in ['_frame_count','_dur','_first_ocr','_last_ocr','_total_ocr']]
        
        for col in columns:
            df[col] = self.impute_Median_col(df,col)
        return df
    
    def imputeConstant(self,df,columns=None,constant=0):
        if isinstance(columns,list) == False:
            return df.fillna(constant)
        return df[columns].fillna(constant)
    
    def normalize(self,df,axis):
        return normalize(df,axis=1)
    
    def overSampling(self,xDF,yDF,col_as_label = 'length_of_ad_15',each_Col = False,rmCols = ['Video Name','length_of_ad_15'],size=400,random_state=None):
        sm = SMOTE(sampling_strategy = {0:size//2,1:size//2},random_state=random_state)
        video_name= pd.DataFrame()
        video_name['Video Name'] = xDF['Video Name']
        temp_inp_df = pd.concat([xDF,yDF],1)
        labels = temp_inp_df.loc[:,col_as_label]
        if each_Col == False:
            temp_inp_df = temp_inp_df.drop(rmCols,1)
        else:
            pass
#         print((xDF['Video Name']==yDF['Video Name']).sum(),temp_inp_df.isna().sum().sum())
#         print(temp_inp_df.isna().sum())
        X_res, y_res = sm.fit_resample(temp_inp_df,labels)
        
        resamp_inputs_df = pd.DataFrame(X_res,columns=temp_inp_df.columns)
        resamp_target_df = pd.DataFrame(y_res,columns=[col_as_label])
        resamp_data_df = pd.concat([resamp_inputs_df,resamp_target_df],1)
        # print(video_name,video_name.shape)
        # resamp_data_df['Video Name'] = video_name['Video Name']
        # resamp_inputs_df = resamp_data_df[xDF.columns]
        resamp_inputs_df = resamp_data_df[xDF.drop('Video Name',1).columns]
        resamp_target_df = resamp_data_df[yDF.columns]
        
        return resamp_inputs_df,resamp_target_df