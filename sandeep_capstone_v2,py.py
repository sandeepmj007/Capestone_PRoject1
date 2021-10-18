import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
import numpy.random as nr
import pandas as pd
import math
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from scipy.spatial.distance import euclidean
import warnings
import seaborn as sns
from sklearn.metrics import roc_auc_score 


read=1;
primary_analysis=0;
define_variables=1;
visualization=1;
sample_diff=0;
feature_transformation=1;
feature_engineering=0;
scaling=1;
oversampling=0
compare=0;
stratified_sample=1 ;
pca=0;
noraml_train_classification=0;#should be trained directly without any preprocessing.
stratied_train_classification=1;# should be trained with feature_transformation,scaling,stratifed_sample,

if read==1:
 filepath = "D:\Imarticus learning\mlcp\input\heart_failure_dataset.csv"
 y_name = 'DEATH_EVENT'
 dtype_file = "capstone1_dtype_analysis.txt"
 df = pl.read_data(filepath)
 
 
 
if primary_analysis==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values, categorical to ordinal numbers
    df_h = df.head()
    with open(dtype_file, "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) + 
                    ", missing: " + str(line3)
            + "\n\n" + "-----------------"+"\n")
 
 
 
if define_variables==1:

    y = df[y_name]
    x = df.drop([y_name],axis=1)


if visualization==1:
  for col in df.columns:
    sns.set_style('whitegrid')
    sns.boxplot(df['DEATH_EVENT'],df[col])
    plt.show()


if sample_diff==1:   
  df["DEATH_EVENT"].value_counts().plot.bar()
  plt.xlabel('Death_event')
  plt.ylabel('count')
  plt.show()
  sample_diff, min_y, max_y = pl.bias_analysis(df,y_name)
  print("sample diff:", sample_diff)
  print("sample ratio:", min_y/max_y)
  print(df[y_name].value_counts())

if feature_transformation==1:
 
    list_=df['age']
    age_grp = {1:range(1,21),2:range(21,41),3:range(41,60),4:range(60,101)}
    x['age']= [key for v in list_
          for key,val in age_grp.items() if v in val]


if feature_engineering==1:

    x['disease']=df['anaemia'].astype(str)+df['sex'].astype(str)+df['diabetes'].astype(str)+df['high_blood_pressure'].astype(str)+df['smoking'].astype(str)
    x=x.drop(['anaemia','sex','diabetes','high_blood_pressure','smoking'],axis=1)
    
    x['disease']=pl.label_encode(x['disease'])
    #df_cat=pl.onehot_encode(df_cat)
 
    
if scaling==1: 
    df_cat=x[['anaemia','sex','diabetes','high_blood_pressure','smoking']]
    df_num= x.drop(['anaemia','sex','diabetes','high_blood_pressure','smoking'],axis=1)
    #df_cat=x[['age','disease']]
    #df_num= x.drop(['age','disease'],axis=1)

    df_num=pl.minmax_normalization(df_num)
    x = pl.join_num_cat(df_num,df_cat) 


if oversampling==1:
     x,y = pl.oversampling(x,y)
     print(x.shape); print(y.value_counts())


if pca==1:
    x = pl.reduce_dimensions(x, 6); #print(x.shape)
    x = pd.DataFrame(x)
    print("transformed x:")
    print(x.shape); print("")
    
if compare==1:
    
    #compare models on sample
    n_samples = 5000
    df_temp = pd.concat((x,y),axis=1)
    df_sample = pl.stratified_sample(df_temp, y_name, n_samples)
    print("stratified sample:"); print(df_sample[y_name].value_counts())
    #df_sample = df_temp
    #print("Non-Stratified sample:");print(df_sample[y_name].value_counts())
    y_sample = df_sample[y_name]
    x_sample = df_sample.drop([y_name],axis=1)
    
    model_meta_data = pl.compare_models(x_sample, y_sample, 111,)
    best_model_id = model_meta_data['best_model'][0]
    best_model = cl.get_models()[best_model_id]; print(best_model)



nr.seed(100)
if stratified_sample==1:
 
 
  n_samples = 5000
  df_temp = pd.concat((x,y),axis=1)
  df_sample = pl.stratified_sample(df_temp, y_name, n_samples)
  print("stratified sample:"); print(df_sample[y_name].value_counts())

  best_model = cl.GradientBoostingClassifier()
  pl.kfold_cross_validate(best_model, x, y,171)

  y_new=df_sample[y_name]
  x_new=df_sample.drop([y_name],axis=1)



if noraml_train_classification==1:
   #model1 = cl.RandomForestClassifier(random_state=169,)
   model1 = cl.RandomForestClassifier(random_state=169,max_depth=4,class_weight={0:0.75,1:1})
   trained_model = pl.clf_train_test(model1,x,y,100,"cap_trail",pred_th=None)

if stratied_train_classification==1:

   #model1 = cl. DecisionTreeClassifier(random_state=169,max_depth=2,max_leaf_nodes=3)
   #model1= cl.LogisticRegression(random_state=169,class_weight={0:0.7,1:1},)
   model1 = cl.RandomForestClassifier(random_state=169,max_depth=3,class_weight={0:0.7,1:1},)
   #model1 = cl.GradientBoostingClassifier(random_state=169,max_depth=1,learning_rate=0.2,)
   trained_model = pl.clf_train_test(model1,x_new,y_new,100,"cap_trail",pred_th=None)
   

  
