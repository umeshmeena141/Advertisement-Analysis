import pandas as pd
import numpy as np
import copy,re
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

from dataParsing import dataParsing
class featureEngineering(dataParsing):
    
    def __init__(self):
        super().__init__()
        self.feature_columns = ['Video Name',
       'length_of_ad_15', 'ad_type_tv', 'Female_frame_count',
       'Male_frame_count', 'Female_dur', 'Male_dur',
       'Age: 5-15_frame_count', 'Age: 0-5_frame_count',
       'Age: 15-24_frame_count', 'Age: 25-34_frame_count',
       'Age: 35-44_frame_count', 'Age: 45-60_frame_count',
       'Age: 5-15_dur', 'Age: 0-5_dur', 'Age: 15-24_dur',
       'Age: 25-34_dur', 'Age: 35-44_dur', 'Age: 45-60_dur',
       'Angry_frame_count', 'Disgust_frame_count', 'Fear_frame_count',
       'Happiness_frame_count', 'Neutral_frame_count',
       'Sadness_frame_count', 'Surprise_frame_count', 'Angry_dur',
       'Disgust_dur', 'Fear_dur', 'Happiness_dur', 'Neutral_dur',
       'Sadness_dur', 'Surprise_dur', 'African_frame_count',
       'Caucasian_frame_count', 'Eastasian_frame_count',
       'Latino_frame_count', 'African_dur', 'Caucasian_dur',
       'Eastasian_dur', 'Latino_dur', 'microsoft', 'microsoft_dur',
       'microsoft_first_ocr', 'microsoft_last_ocr', 'microsoft_total_ocr',
       'Intel_dur', 'Intel_first_ocr', 'Intel_last_ocr',
       'Intel_total_ocr', 'ASUS T102', 'Dell PCs', 'Dell XPS',
       'Dell XPS 13', 'HP Spectre x360', 'HP Spectre x360, Windows',
       'Lenovo PCs', 'Lenovo Yoga 720', 'Lenovo Yoga 910', 'MacBook Air',
       'MacBook Pro', 'Macbook Air', 'Microsoft Surface', 'Suface Laptop',
       'Surface Book', 'Surface Go', 'Surface Laptop', 'Surface Pen',
       'Surface Pro', 'Surface Pro 4',
       'Surface Pro 4, Apple_MacBookAir13',
       'Surface Pro 4, HP Spectre x360', 'Surface Pro 4, MacBook Air',
       'Surface Pro 4, MacBook Air13, MacBook Air',
       'Surface Pro 4, Surface Book', 'Surface Pro 4, Windows 10',
       'Surface Pro 6', 'Surface laptop', 'Surface pro 4',
       'The new Surface Pro', 'The surface Laptop', 'Windows',
       'Windows 10', 'Windows 10 PC', 'Windows 10 PCs',
       'Windows 10, HP_Spectre', 'Windows 10, Windows', 'Yoga',
       'ASUS T102_dur', 'Dell PCs_dur', 'Dell XPS_dur', 'Dell XPS 13_dur',
       'HP Spectre x360_dur', 'HP Spectre x360, Windows_dur',
       'Lenovo PCs_dur', 'Lenovo Yoga 720_dur', 'Lenovo Yoga 910_dur',
       'MacBook Air_dur', 'MacBook Pro_dur', 'Macbook Air_dur',
       'Microsoft Surface_dur', 'Suface Laptop_dur', 'Surface Book_dur',
       'Surface Go_dur', 'Surface Laptop_dur', 'Surface Pen_dur',
       'Surface Pro_dur', 'Surface Pro 4_dur',
       'Surface Pro 4, Apple_MacBookAir13_dur',
       'Surface Pro 4, HP Spectre x360_dur',
       'Surface Pro 4, MacBook Air_dur',
       'Surface Pro 4, MacBook Air13, MacBook Air_dur',
       'Surface Pro 4, Surface Book_dur', 'Surface Pro 4, Windows 10_dur',
       'Surface Pro 6_dur', 'Surface laptop_dur', 'Surface pro 4_dur',
       'The new Surface Pro_dur', 'The surface Laptop_dur', 'Windows_dur',
       'Windows 10_dur', 'Windows 10 PC_dur', 'Windows 10 PCs_dur',
       'Windows 10, HP_Spectre_dur', 'Windows 10, Windows_dur',
       'Yoga_dur', 'Windows 10_first_ocr', 'Windows 10_last_ocr',
       'Windows 10_total_ocr', 'Lenovo Yoga 910_first_ocr',
       'Lenovo Yoga 910_last_ocr', 'Lenovo Yoga 910_total_ocr',
       'Yoga_first_ocr', 'Yoga_last_ocr', 'Yoga_total_ocr',
       'Surface Pro 4_first_ocr', 'Surface Pro 4_last_ocr',
       'Surface Pro 4_total_ocr',
       'Surface Pro 4, Apple_MacBookAir13_first_ocr',
       'Surface Pro 4, Apple_MacBookAir13_last_ocr',
       'Surface Pro 4, Apple_MacBookAir13_total_ocr',
       'Surface Pro 4, MacBook Air13, MacBook Air_first_ocr',
       'Surface Pro 4, MacBook Air13, MacBook Air_last_ocr',
       'Surface Pro 4, MacBook Air13, MacBook Air_total_ocr',
       'Surface Book_first_ocr', 'Surface Book_last_ocr',
       'Surface Book_total_ocr', 'MacBook Pro_first_ocr',
       'MacBook Pro_last_ocr', 'MacBook Pro_total_ocr',
       'Microsoft Surface_first_ocr', 'Microsoft Surface_last_ocr',
       'Microsoft Surface_total_ocr', 'MacBook Air_first_ocr',
       'MacBook Air_last_ocr', 'MacBook Air_total_ocr',
       'Surface Pro 4, Surface Book_first_ocr',
       'Surface Pro 4, Surface Book_last_ocr',
       'Surface Pro 4, Surface Book_total_ocr', 'ASUS T102_first_ocr',
       'ASUS T102_last_ocr', 'ASUS T102_total_ocr',
       'Windows 10, HP_Spectre_first_ocr',
       'Windows 10, HP_Spectre_last_ocr',
       'Windows 10, HP_Spectre_total_ocr', 'HP Spectre x360_first_ocr',
       'HP Spectre x360_last_ocr', 'HP Spectre x360_total_ocr',
       'Surface Pro 4, Windows 10_first_ocr',
       'Surface Pro 4, Windows 10_last_ocr',
       'Surface Pro 4, Windows 10_total_ocr',
       'Surface Pro 4, HP Spectre x360_first_ocr',
       'Surface Pro 4, HP Spectre x360_last_ocr',
       'Surface Pro 4, HP Spectre x360_total_ocr',
       'Windows 10, Windows_first_ocr', 'Windows 10, Windows_last_ocr',
       'Windows 10, Windows_total_ocr', 'Windows_first_ocr',
       'Windows_last_ocr', 'Windows_total_ocr',
       'HP Spectre x360, Windows_first_ocr',
       'HP Spectre x360, Windows_last_ocr',
       'HP Spectre x360, Windows_total_ocr', 'Dell XPS 13_first_ocr',
       'Dell XPS 13_last_ocr', 'Dell XPS 13_total_ocr',
       'Surface Pen_first_ocr', 'Surface Pen_last_ocr',
       'Surface Pen_total_ocr', 'Macbook Air_first_ocr',
       'Macbook Air_last_ocr', 'Macbook Air_total_ocr',
       'Windows 10 PC_first_ocr', 'Windows 10 PC_last_ocr',
       'Windows 10 PC_total_ocr', 'Windows 10 PCs_first_ocr',
       'Windows 10 PCs_last_ocr', 'Windows 10 PCs_total_ocr',
       'Surface Pro 4, MacBook Air_first_ocr',
       'Surface Pro 4, MacBook Air_last_ocr',
       'Surface Pro 4, MacBook Air_total_ocr', 'Surface pro 4_first_ocr',
       'Surface pro 4_last_ocr', 'Surface pro 4_total_ocr',
       'Lenovo Yoga 720_first_ocr', 'Lenovo Yoga 720_last_ocr',
       'Lenovo Yoga 720_total_ocr', 'Dell PCs_first_ocr',
       'Dell PCs_last_ocr', 'Dell PCs_total_ocr',
       'Surface Laptop_first_ocr', 'Surface Laptop_last_ocr',
       'Surface Laptop_total_ocr', 'The new Surface Pro_first_ocr',
       'The new Surface Pro_last_ocr', 'The new Surface Pro_total_ocr',
       'Surface Pro_first_ocr', 'Surface Pro_last_ocr',
       'Surface Pro_total_ocr', 'Suface Laptop_first_ocr',
       'Suface Laptop_last_ocr', 'Suface Laptop_total_ocr',
       'Surface laptop_first_ocr', 'Surface laptop_last_ocr',
       'Surface laptop_total_ocr', 'The surface Laptop_first_ocr',
       'The surface Laptop_last_ocr', 'The surface Laptop_total_ocr',
       'Surface Go_first_ocr', 'Surface Go_last_ocr',
       'Surface Go_total_ocr', 'Surface Pro 6_first_ocr',
       'Surface Pro 6_last_ocr', 'Surface Pro 6_total_ocr',
       'Dell XPS_first_ocr', 'Dell XPS_last_ocr', 'Dell XPS_total_ocr',
       'Lenovo PCs_first_ocr', 'Lenovo PCs_last_ocr',
       'Lenovo PCs_total_ocr','Laptop','Laptop_dur','Laptop_first_ocr',
        'Laptop_last_ocr','Laptop_total_ocr','Tablet','Tablet_dur','Tablet_first_ocr',
        'Tablet_last_ocr','Tablet_total_ocr','Mobile','Mobile_dur','Mobile_first_ocr',
        'Mobile_last_ocr','Mobile_total_ocr','PC','PC_dur','PC_first_ocr',
        'PC_last_ocr','PC_total_ocr'
                               
                               ]
        self.target_columns = self.targets
        
        self.brands_columns=["Intel","Apple","Asus","Hp","Dell"]
        self.related_brand = "microsoft"
    
        
    def addUnrelatedBrands(self,X,brands_columns=None):
        if isinstance(brands_columns,list) == False:
            brands_columns = self.brands_columns
        for param in ['_dur','_first_ocr','_last_ocr','_total_ocr']:
            unrelated_columns = [col+param for col in brands_columns]
            X['Unrelated'+param] = X[X.loc[:,unrelated_columns].columns].sum(axis=1)
            X = X.drop(unrelated_columns,axis=1)
        
        return X
    
    def nullPer(self,df):
        """Quantifies missing values"""
#         try:        
        return(df.isnull().mean()*100)
#         except:
#             pass
    
    def renameRelatedBrand(self,X,brand_name=None):
        if isinstance(brand_name,str) == False:
            brand_name = self.related_brand
        columns = [brand_name+col for col in ['','_dur','_first_ocr','_last_ocr','_total_ocr']]
        X.rename(columns=dict(zip(columns, ['Related_Brand_frame_count','Related_Brand_dur', 'Related_Brand_first_ocr',
            'Related_Brand_last_ocr', 'Related_Brand_total_ocr'])),inplace=True)
        return X
    
    def addNewFeature(self,X,columns=None,_type='per',new_column_name=None,suffix='_dur'):
        if isinstance(columns,list):
            if _type =='sum':
                X[new_column_name] = X[X.loc[:,columns].columns].sum(axis=1)
            else:
                for feature in columns:
                    index_1,index_2 = feature[0].rfind(suffix),feature[1].rfind(suffix)
                    split_1,split_2 = feature[0][:index_1],feature[1][:index_2]
                    try:
                        X.insert(len(X.columns)-12,split_1+"_"+split_2+suffix,X[feature[1]]/X[feature[0]])
                    except:
                        X[split_1+"_"+split_2+suffix] = X[feature[1]]/X[feature[0]]
        else:
            print("Please provide columns names and new column name")
        return copy.deepcopy(X)
    
    def clean_data(self,X,Y):
        Y =  Y.dropna(how='all') 
        return X.loc[:,self.feature_columns],Y
    
#     def correlation_matrix(self,X):
#         self.utils.plotCorr(X)
        
    def impute_Median_col(self,df, x):
        """Imputes median - treatement for missing values in Pandas series"""
#         try:
        df[x] = df[x].fillna(df[x].median())
    
        return df[x]
#         except:
#             return df[x]
#             pass

    def correlation(self,df,per=0.2,savefig = False):
        corr = df.corr()
        links = corr.stack().reset_index()
        links.columns = ['var1', 'var2','value']

        # Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
        links_filtered=links.loc[ (links['value'] > per) & (links['var1'] != links['var2']) ]

        # Build your graph
        # links_filtered = links
        G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
        # color_list = 
        values=[]
        for e in G.edges():
            Y = links_filtered[links_filtered['var1']==e[0]]
            Z = Y[Y['var2']==e[1]]
            values.append(Z['value'].values[0])

        minima = min(values)
        maxima = max(values)

        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap="RdYlGn")    

        for i,val in enumerate(values):
            values[i] = matplotlib.colors.to_hex(mapper.to_rgba(val))
        # color_map = le.fit_transform(values)
        plt.figure(figsize=(10,10))
        nx.draw_circular(G, with_labels=True,node_color="skyblue",edge_color=values,font_size=8)
        if savefig:
            plt.savefig('correlation.png')
        plt.show()