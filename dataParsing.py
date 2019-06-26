import pandas as pd
import numpy as np
import copy,re

class dataParsing:
    
    def __init__(self):
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        self.data = pd.DataFrame()
        self.labels = False
        self.targets = [
            'Unaided_Branding', 'Brand_Cues__Mean','Aided_Branding__Mean',
           'Active_Involvement__Mean','New_Information__Mean', 'Enjoyment__Mean',
           'Brand_Appeal__Mean', 'Understanding__Mean','Relevance_of_Information__Mean',
           'Credibility_of_Information__Mean','Brand_Difference__Mean',
           'Interest_peak','Interest_mean_score','Purchase_intent','Persuasion_mean',
           'Persuasion_very_likely','Interest_peak_frames']
        
    def separate_X_Y(self):
        for tr in self.targets:
            for col in self.data.columns:
                if tr in col:
                    self.Y[tr] = self.data[col]
                else:
                    self.X[col] = self.data[col]

        return self.X,self.Y
            
    
    def load_data(self,path_to_file = None,sheet_name="Data",labels=True):
        
        if path_to_file:
            self.data = pd.read_excel(path_to_file,sheet_name=sheet_name)
            X,Y = self.separate_X_Y()
            return X,Y
        else:
            print("Please provide path to file\n")
            return None
            
    def load_Y(self,path_to_file=None,video_name=False,column_name="same"):
        self.Y = pd.DataFrame()
        if path_to_file:
            Y_data = pd.read_excel(path_to_file,sheet_name=0,index_col=0)
            if column_name=="diff":
                for tr in self.targets:
                    for col in Y_data.columns:
                        if re.sub("[^a-zA-Z]", "",tr) in re.sub("[^a-zA-Z]", "",col):
                            self.Y[tr] = Y_data[col]
            else:
                for col in Y_data.columns:
                    self.Y[col] = Y_data[col]

            if video_name:
                self.Y["Video Name"] = Y_data["Video Name"]
        else:
            print("Please provide path to file\n")

        return copy.deepcopy(self.Y)
                        
    def isLabels(self):
        if self.Y.shape[0]!=0 :
            return True
        return False
    