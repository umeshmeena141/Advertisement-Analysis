from itertools import groupby
import itertools
import pandas as pd
import numpy as np
from copy import deepcopy

class Agg_Frame_Data:
    
    def __init__(self,df):
        self.agg_inp = pd.DataFrame()
        self.agg_inp['Video Name'] = df['Video Name'].unique()
        self.inp_df = df
        self.unstack_cols = {'Ethnicity':['African','Caucasian','Eastasian','Latino'],
                            'Emotion': ['Angry','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'],
                             'Country': ['US','UK','AU'],
                             'Length of the Ad':[15],'AD Type: TV/DIGITAL':['TV']
                            }
        self.cat_cols ={'Length of the Ad':['length_of_ad_15'],'AD Type: TV/DIGITAL':['ad_type_tv']}
        self.agg_inp = self.agg_inp.set_index(df['Video Name'].unique())
        
    def countryEncoding(self,x):
        for c in ['Country_AU','Country_US','Country_UK']:
            if x[c]>0:
                x[c] = 1
            else:
                x[c] =0
        return x
        
    def _unstack(self,col_name,other=[],binary=False):
        print("Collecting "+col_name+" data")
        temp_df = pd.DataFrame()
        col_df = self.inp_df[[col_name,'Video Name']].groupby([col_name,'Video Name']).size().unstack().unstack()
        if len(other)==0:
            col_df = col_df.unstack()[self.unstack_cols[col_name]]
        else:
            col_df = col_df.unstack()[other]
        
        if not binary:
            if col_name == 'Country':
                cols = ['Country_'+col for col in self.unstack_cols[col_name]]
                temp_df[cols]= col_df[self.unstack_cols[col_name]]
                temp_df = temp_df.set_index(col_df.index.values)
                self.agg_inp = pd.concat((self.agg_inp,temp_df),1)
                self.agg_inp = self.agg_inp.apply(self.countryEncoding,1)
            else:
                cols = [col+'_frame_count' for col in self.unstack_cols[col_name]]
                temp_df[cols] = col_df[self.unstack_cols[col_name]]
                temp_df = temp_df.set_index(col_df.index.values)
                self.agg_inp = pd.concat((self.agg_inp,temp_df),1)
                for col in self.unstack_cols[col_name]:
                    new_col = col+'_dur'
                    self.agg_inp[new_col] = self.agg_inp[col+'_frame_count']*0.5
        else:
            temp_df[self.cat_cols[col_name]] = col_df[self.unstack_cols[col_name]]
            temp_df = temp_df.set_index(col_df.index.values)
            self.agg_inp = pd.concat((self.agg_inp,temp_df),1)
            self.agg_inp.loc[self.agg_inp[self.cat_cols[col_name][0]].isnull(),self.cat_cols[col_name]] = 0
            self.agg_inp.loc[self.agg_inp[self.cat_cols[col_name][0]]!= 0,self.cat_cols[col_name]] = 1
        
        return deepcopy(self.agg_inp)
    
    def separate_age_gender(self):
        print("Separating Age and Gender")
        self.inp_df['Age'] = self.inp_df['Age-Gender']
        self.inp_df['Gender'] = self.inp_df['Age-Gender']
        for i in range(len(self.inp_df['Age-Gender'])):
            if len(self.inp_df['Age'][i].split(',')) == 2:
                self.inp_df.loc[i,'Age'] = self.inp_df.loc[i,'Age-Gender'].split(',')[0]
                self.inp_df.loc[i,'Age'] = (int(self.inp_df.Age[i].split(":")[1].split("-")[0]) + int(self.inp_df.Age[i].split(":")[1].split("-")[1].replace("'","")))/2
                self.inp_df.loc[i,'Gender'] = self.inp_df.loc[i,'Age-Gender'].split(',')[1]
            else:
                self.inp_df.loc[i,'Age'] = float('nan')
        
        self.inp_df.Age = inp_df.Age.astype('float')
        age_df_vid = self.inp_df[['Video Name','Age']].groupby(['Video Name','Age']).size().unstack()
        self.agg_inp[['Age: 0-5_frame_count','Age: 5-15_frame_count','Age: 15-24_frame_count','Age: 25-34_frame_count','Age: 35-44_frame_count','Age: 45-60_frame_count']]=age_df_vid[[2.5,7.5,19.5,29.5,39.5,52.5]]
        gender_df_vid = inp_df[['Gender','Video Name']].groupby(['Video Name','Gender']).size().unstack(fill_value =0).drop('No Face Found',1)
        self.agg_inp[["Female_frame_count","Male_frame_count"]] = gender_df_vid[[" 'Female']"," 'Male']"]]
        cols =["Female_frame_count","Male_frame_count",'Age: 0-5_frame_count','Age: 5-15_frame_count','Age: 15-24_frame_count','Age: 25-34_frame_count','Age: 35-44_frame_count','Age: 45-60_frame_count']
        for col in cols:
            index = col.find('_frame_count')
            new_col = col[:index]+'_dur'
            self.agg_inp[new_col] = self.agg_inp[col]*0.5
        
        return deepcopy(self.agg_inp)
    
    def lower(self,row,col):
        row[col] = row[col].lower()
        return row
    
    def add(self,x):
        return x[0]+x[1]
    
    def agg_logo_text_data(self,columns_):
        logo_df = pd.DataFrame()
        for c in columns_:
            print(c)
            if c=='Microsoft-Logo':
                self.inp_df[self.inp_df[c].isna()==False] = self.inp_df[self.inp_df[c].isna()==False].apply(lambda x: self.lower(x,c),1)  
            temp_logo_df = self.inp_df[[c,'Video Name']].groupby(['Video Name',c]).size().unstack(fill_value = 0)
            #else:
            #    logo_df = pd.concat([logo_df,inp_df[[c,'Video Name']].groupby(['Video Name',c]).size().unstack(fill_value = 0)],axis=1)

            ## Duration
            dur_df = temp_logo_df*0.5
            var_cols =temp_logo_df.columns.values
            #print(var_cols)
            dur_df.columns = temp_logo_df.columns.values + '_dur'

            if len(logo_df) == 0:
                logo_df = pd.concat([temp_logo_df,dur_df],axis=1)
            else:
                frame_dur_df = pd.concat([temp_logo_df,dur_df],axis=1)
                logo_df = pd.concat([logo_df,frame_dur_df],axis=1)

            ## Occurences


            vids = temp_logo_df.index
        #     for unq_val in range(len(inp_df.loc[:,c].unique())):

        #         if str(inp_df.loc[:,c].unique()[unq_val]) != 'nan':
        #             #print(unq_val)
            unq_val = 1
            ocr_list = list(itertools.product(var_cols,['_first_ocr','_last_ocr','_total_ocr'])) 
            ocr_cols = map(self.add,ocr_list)
            ocr_df = pd.DataFrame(columns = list(ocr_cols))
            for v in range(len(vids)):
        #                 print(inp_df.loc[:,c].unique()[unq_val])
        #                 print(len(inp_df.loc[inp_df['Video Name'] == vids[v],c].unique()))
                first_ocr =[None]*len(var_cols)
                last_ocr =[None]*len(var_cols)
                total_ocr =[None]*len(var_cols)

                if len(self.inp_df.loc[self.inp_df['Video Name'] == vids[v],c].unique()) >1:
                    #print(inp_df.loc[inp_df['Video Name'] == vids[v],c].unique())
                    for k,var in enumerate(var_cols):
                        ocr_op =   [i for i,x in enumerate(self.inp_df.loc[inp_df['Video Name'] == vids[v],c]) if x==var]
        #                 print(ocr_op)

                        if len(ocr_op) > 0:
                            ocr_df.loc[v,str(var_cols[k]+'_first_ocr')] = ocr_op[0]*0.5
                            ocr_df.loc[v,str(var_cols[k]+'_last_ocr')] = ocr_op[-1]*0.5
                            ocr_df.loc[v,str(var_cols[k]+'_total_ocr')] = len([x for x in list(np.diff(ocr_op)) if x != 1]) +1
                        else:
                            ocr_df.loc[v,str(var_cols[k]+'_first_ocr')] = None
                            ocr_df.loc[v,str(var_cols[k]+'_last_ocr')] = None
                            ocr_df.loc[v,str(var_cols[k]+'_total_ocr')] = None

                else:
                    ocr_df.loc[v,str(var_cols[0]+'_first_ocr')] = None
                    ocr_df.loc[v,str(var_cols[0]+'_last_ocr')] = None
                    ocr_df.loc[v,str(var_cols[0]+'_total_ocr')] = None
            ocr_df.index = vids
            logo_df = pd.concat([logo_df,ocr_df],axis=1)
                #print(logo_df)
        return logo_df

    def agg_column_data(self,columns):
        logo_df = self.agg_logo_text_data(columns)
        self.agg_inp = pd.concat((self.agg_inp,logo_df),axis=1)
        return deepcopy(self.agg_inp)