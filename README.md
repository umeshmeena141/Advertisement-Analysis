There are 7 folders namely:
    - Agg_target_variables
    - Classification
    - Data
    - Data_Preprocessing_IPYNB
    - Factor_Analysis
    - Model Class
    - Regression

<!-- If there is any sort of error in loading or working of any function of code please go through Model class and some common error parts below -->

How To RUN IPYNB files:-
    - Provide correct path to model class at the start of ipynb file. 
      Use sys.path.append($PATH_TO_MODEL_CLASS) to give relative path.

Agg_target_variables:- 
    - This folder contains files which are used to aggregate data from frame levels. New Target 
      Variables.ipynb will create new target variables like Interest peak, Persuasion score, Persuasion Intent and all most likely variables from frame level data.
    - Data Preprocessing-V3 file can be used to aggregate features. It has to be run till Frame Level       Analysis class for aggregation. The code for aggregation of data is also separated in a python file   named Agg_Frame_Data.py

Classification:-
    - This folder contains three other folders which are Classification_IPYNB, Interest Trace,              Version_1_class.
    - All the IPYNB notebooks for classification task has been put in this folder.
    - All classification tasks IPYNB for Interest Trace is kept in the second folder.
    - Version_1_class folder contains analysis done for target variables, filtered by software and          surface.

Data:-
    - All the Data files which are required in IPYNB files are stored in this folder.

Data_Preprocessing_IPYNB:-
    - There are six different files in this folder. First three file starting with Data Preprocessing are   the first the version files for training before aggregation of model class. It has all model class    written in same IPYNB file. Latest version also contains correlation graph in circular layout.
    - Second set of files are Ad_Opt_Analysis which are cleaned versions of Data Preprocessing file and     used for grid search of parameters for different models. Stacked Model is also experimented in        latest version.

Factor Analysis:-
    - It contains two different versions of IPYNB files, one of them is experimentation of ANOVA and PCA    along with factor analysis.
    - Latest file extend previous version to execute on the input dataframe.


Model Class:-
    - This folder has all python files/classes that has been used in IPYNB notebooks.
    - Model.py: It contains model class which has training of model, plotting of decision tree, grid        search for hyper parameter tuning and plotting of training and learning curves.
      It can be used for both classification and regression task and for any of three models - RF, DT and xgb. The subclass to the class is Evaluate class and Data preprocessing class is initialised here.
    - Evaluate.py: It evaluates the models for different evaluation metrices on test set. It can also be    used to plot and print feature importance in a sorted manner. It also has function, which can         directly provide you shap analysis for important features.
    - DataPreprocessing.py: Preprocess, clean and oversampled data from imblanced class.
    - featureEngineering.py: It has list of all the features which has to be used finally before cleaning   of data. It removes null columns, also plot correlation graph.
    - dataParsing: Mostly loading of data. You can change target variable's names here.
    - Agg_Frame_Data.py: Go through first folder description, third para.

Regression:-
    - IPYNB files for Interest Trace and other target variables for regression tasks.
    - Different version of analysis has been provided.
    - Latest version is WIth_new_target_variables, which is filtered for surface and software variables.
    - Use Version_6_new_variables.ipynb to continue working with.

Common Errors:-
    - During Oversampling, make sure only input df has Video Name as one of the column.
    - Classification and Regression xgb training can produce errors due to metrics, please refer xgb docs   and replace the metrics in Model.py for xgb part with suitable metrics.
    - Please use updated version imblearn library for SMOTE.
    - Change list of variables in featureEngineering.py for removing some unwanted feature columns.
