#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


zhengqi_train=pd.read_csv('zhengqi_train.txt',sep='\t')
zhengqi_test=pd.read_csv('zhengqi_test.txt',sep='\t')
print(zhengqi_train.head())
# zhengqi test donot have output
print(zhengqi_test.head())
print(zhengqi_train.shape)
print(zhengqi_test.shape)


# In[ ]:


# target is our purpose, so firstly plot it to see whether there exis outliers
plt.figure(figsize=(8,6))
plt.scatter(range(zhengqi_train.shape[0]),np.sort(zhengqi_train['target']))
plt.xlabel('index')
plt.ylabel('value_of_target')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(zhengqi_train['target'],bins=50,kde=True)
plt.xlabel('value_of_target')
# a little bit skewed


# In[ ]:


# generate a datafrmae containing the index of column and the data type of each column
feature_types=zhengqi_train.dtypes.reset_index()
# rename column name of feature_types whose type is a dataframe
feature_types.columns=['name_of_column','data_type']
print(feature_types)
feature_types.groupby('data_type').aggregate('count').reset_index()


# In[ ]:


# check missing values
def check_missing_values_by_column(data):
    assert isinstance(data,pd.DataFrame)
    missing_values=data.isnull().sum(axis=0).reset_index()
    missing_values=missing_values[missing_values.iloc[:,1]>0]
    print(missing_values)

check_missing_values_by_column(zhengqi_train)
# empty,no missing value


# In[ ]:


# check duplicate rows
zhengqi_train[zhengqi_train.drop(['target'],axis=1).duplicated(keep=False)]


# In[ ]:


# plot to compare distribution of train set and test set
def plot_each_column(zhengqi_train,zhengqi_test):
    column_names=list(zhengqi_train)
    length=len(column_names)
    print(length)
    for i in range(length):
        if column_names[i]=='target':
            pass
        else:
            plt.figure(figsize=(20,10))
            plt.subplot(1,3,1)
            plt.scatter(range(zhengqi_train.shape[0]),np.sort(zhengqi_train[column_names[i]]))
            plt.scatter(range(zhengqi_test.shape[0]),np.sort(zhengqi_test[column_names[i]]))
            plt.xlabel('index')
            plt.ylabel(column_names[i])
            plt.legend(['train_set','test_set'])

            plt.subplot(1,3,2)
            sns.distplot(zhengqi_train[column_names[i]],bins=50,kde=True)
            sns.distplot(zhengqi_test[column_names[i]],bins=50,kde=True)
            plt.xlabel(column_names[i])
            plt.legend(['train_set','test_set'])

            plt.subplot(1,3,3)
            plt.scatter(x=column_names[i], y='target', data=zhengqi_train)
            plt.xlabel(column_names[i])
            plt.ylabel('target')

            plt.show()
        
# plot_each_column(zhengqi_train,zhengqi_test)


# In[ ]:


"""
第一步，把异常值去掉
"""
# drop outliers that greater than or smaller test set
def drop_points_v1(train_set,test_set):
    column_names=list(train_set)
    for name in column_names[:-1]:
        max_value_test=test_set[name].max()
        min_value_test=test_set[name].min()
#         找train dataset中每个column里最大最小的三个数
        temp_min,temp_max=train_set.nsmallest(1,name),train_set.nlargest(3,name)
        min_value_train=temp_min[name].max()
        max_value_train=temp_max[name].min()
#         print(max_value_train)
        if max_value_train>max_value_test:
            train_set=train_set[train_set[name]<max_value_train]
        if min_value_train<min_value_test:
            train_set=train_set[train_set[name]>min_value_train]
    return train_set

def drop_points_v2(train_set,test_set):
    train_set=train_set.copy()
    column_names=list(train_set)
    for name in column_names[:-1]:
        if name=='V9':
            temp_min=train_set.nsmallest(2,name)
            min_value_train=temp_min[name].max()
            train_set=train_set[train_set[name]>min_value_train]
        if name=='V10':
            train_set=train_set[train_set[name]<3.6]
        if name=='V15':
            temp_min,temp_max=train_set.nsmallest(1,name),train_set.nlargest(3,name)
            min_value_train=temp_min[name].max()
            max_value_train=temp_max[name].min()
            train_set=train_set[train_set[name]<max_value_train]
            train_set=train_set[train_set[name]>min_value_train]
        if name=='V17':
            train_set=train_set[train_set[name]<1.7]
        if name=='V23':
            temp_max=train_set.nlargest(2,name)
            max_value_train=temp_max[name].min()
            train_set=train_set[train_set[name]<max_value_train]
        if name=='V24':
            temp_max=train_set.nlargest(6,name)
            max_value_train=temp_max[name].min()
            train_set=train_set[train_set[name]<max_value_train]
        if name=='V29':
            temp_max=train_set.nlargest(3,name)
            max_value_train=temp_max[name].min()
            train_set=train_set[train_set[name]<max_value_train]
        if name=='V36':
            temp_max=train_set.nlargest(1,name)
            max_value_train=temp_max[name].min()
            train_set=train_set[train_set[name]<max_value_train]
    return train_set

def process_outlier_in_testset(test_set):
    test_set=test_set.copy()
    column_names=list(test_set)
    for name in column_names:
        if name=='V21':
            temp_min=test_set.nsmallest(2,name)
            min_value=temp_min[name].max()
            print(min_value)
            index_mask=(test_set[name]<=min_value)
            temp_min[name]=temp_min[name]*0.6
            test_set[index_mask]=temp_min
        if name=='V35':
            temp_min=test_set.nsmallest(1,name)
            min_value=temp_min[name].max()
            index_mask=(test_set[name]<=min_value)
            temp_min[name]=-5.9
            test_set[index_mask]=temp_min
    return test_set

zhengqi_test_after_change_points=process_outlier_in_testset(zhengqi_test)
zhengqi_train_after_drop_points=drop_points_v2(zhengqi_train,zhengqi_test_after_change_points)

print(zhengqi_train_after_drop_points.shape)
print(zhengqi_test_after_change_points.shape)
# plot_each_column(zhengqi_train_after_drop_points,zhengqi_test_after_change_points)


# In[ ]:


plot_each_column(zhengqi_train_after_drop_points,zhengqi_test_after_change_points)


# In[ ]:


"""
转换装箱一些变量
"""
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
enc=LabelEncoder()

# def transform_categorical_columns(list_of_columns,train_sest,test_set):
#     for name in list_of_columns:

num_cut=[-7+0.5*i for i in range(20)]
group_name=[str(i) for i in range(19)]
zhengqi_train_after_drop_points["V9cut"]=pd.cut(zhengqi_train_after_drop_points["V9"],num_cut,labels=group_name)
zhengqi_test_after_change_points["V9cut"]=pd.cut(zhengqi_test_after_change_points["V9"],num_cut,labels=group_name)
zhengqi_train_after_drop_points['V9cutlabel']=enc.fit_transform(zhengqi_train_after_drop_points['V9cut'])
zhengqi_test_after_change_points['V9cutlabel']=enc.fit_transform(zhengqi_test_after_change_points['V9cut'])
zhengqi_train_after_drop_points = zhengqi_train_after_drop_points.drop(["V9", "V9cut"], axis=1)
zhengqi_test_after_change_points=zhengqi_test_after_change_points.drop(["V9", "V9cut"], axis=1)

num_cut_v23=[-6+0.5*i for i in range(17)]
group_name_v23=[str(i) for i in range(16)]
zhengqi_train_after_drop_points["V23cut"]=pd.cut(zhengqi_train_after_drop_points["V23"],num_cut_v23,labels=group_name_v23)
zhengqi_test_after_change_points["V23cut"]=pd.cut(zhengqi_test_after_change_points["V23"],num_cut_v23,labels=group_name_v23)
zhengqi_train_after_drop_points['V23cutlabel']=enc.fit_transform(zhengqi_train_after_drop_points['V23cut'])
zhengqi_test_after_change_points['V23cutlabel']=enc.fit_transform(zhengqi_test_after_change_points['V23cut'])
zhengqi_train_after_drop_points = zhengqi_train_after_drop_points.drop(["V23", "V23cut"], axis=1)
zhengqi_test_after_change_points=zhengqi_test_after_change_points.drop(["V23", "V23cut"], axis=1)

num_cut_v24=[-1.5+0.5*i for i in range(9)]
group_name_v24=[str(i) for i in range(8)]
zhengqi_train_after_drop_points["V24cut"]=pd.cut(zhengqi_train_after_drop_points["V24"],num_cut_v24,labels=group_name_v24)
zhengqi_test_after_change_points["V24cut"]=pd.cut(zhengqi_test_after_change_points["V24"],num_cut_v24,labels=group_name_v24)
zhengqi_train_after_drop_points['V24cutlabel']=enc.fit_transform(zhengqi_train_after_drop_points['V24cut'])
zhengqi_test_after_change_points['V24cutlabel']=enc.fit_transform(zhengqi_test_after_change_points['V24cut'])
zhengqi_train_after_drop_points = zhengqi_train_after_drop_points.drop(["V24", "V24cut"], axis=1)
zhengqi_test_after_change_points=zhengqi_test_after_change_points.drop(["V24", "V24cut"], axis=1)

num_cut_v28=[-3+0.5*i for i in range(17)]
group_name_v28=[str(i) for i in range(16)]
zhengqi_train_after_drop_points["V28cut"]=pd.cut(zhengqi_train_after_drop_points["V28"],num_cut_v28,labels=group_name_v28)
zhengqi_test_after_change_points["V28cut"]=pd.cut(zhengqi_test_after_change_points["V28"],num_cut_v28,labels=group_name_v28)
zhengqi_train_after_drop_points['V28cutlabel']=enc.fit_transform(zhengqi_train_after_drop_points['V28cut'])
zhengqi_test_after_change_points['V28cutlabel']=enc.fit_transform(zhengqi_test_after_change_points['V28cut'])
zhengqi_train_after_drop_points = zhengqi_train_after_drop_points.drop(["V28", "V28cut"], axis=1)
zhengqi_test_after_change_points=zhengqi_test_after_change_points.drop(["V28", "V28cut"], axis=1)

num_cut_v35=[-6+0.5*i for i in range(18)]
group_name_v35=[str(i) for i in range(17)]
zhengqi_train_after_drop_points["V35cut"]=pd.cut(zhengqi_train_after_drop_points["V35"],num_cut_v35,labels=group_name_v35)
zhengqi_test_after_change_points["V35cut"]=pd.cut(zhengqi_test_after_change_points["V35"],num_cut_v35,labels=group_name_v35)
zhengqi_train_after_drop_points['V35cutlabel']=enc.fit_transform(zhengqi_train_after_drop_points['V35cut'])
zhengqi_test_after_change_points['V35cutlabel']=enc.fit_transform(zhengqi_test_after_change_points['V35cut'])
zhengqi_train_after_drop_points = zhengqi_train_after_drop_points.drop(["V35", "V35cut"], axis=1)
zhengqi_test_after_change_points=zhengqi_test_after_change_points.drop(["V35", "V35cut"], axis=1)


assert zhengqi_test_after_change_points.shape[1]==(zhengqi_train_after_drop_points.shape[1]-1)
print(zhengqi_test_after_change_points.columns)
print(zhengqi_train_after_drop_points.columns)
print(zhengqi_train_after_drop_points['V9cutlabel'])


# In[ ]:


"""
增加与categorical相关的组合特征
"""
list_of_continous_columns=['V0','V1','V2','V3','V4','V6','V7','V8','V10','V12','V15','V16','V19','V20','V29','V31','V36','V37']

def combine_with_v9(list_of_columns,data_set,test_set):
    data_set=data_set.copy()
    test_set=test_set.copy()
    for name in list_of_columns:
        new_name='V9'+name
        data_set[new_name]=data_set['V9cutlabel']*data_set[name]
        test_set[new_name]=test_set['V9cutlabel']*test_set[name]
    return data_set,test_set


def combine_with_v23(list_of_columns,data_set,test_set):
    data_set=data_set.copy()
    test_set=test_set.copy()
    for name in list_of_columns:
        new_name='V23'+name
        data_set[new_name]=data_set['V23cutlabel']*data_set[name]
        test_set[new_name]=test_set['V23cutlabel']*test_set[name]
    return data_set,test_set

def combine_with_v24(list_of_columns,data_set,test_set):
    data_set=data_set.copy()
    test_set=test_set.copy()
    for name in list_of_columns:
        new_name='V24'+name
        data_set[new_name]=data_set['V24cutlabel']*data_set[name]
        test_set[new_name]=test_set['V24cutlabel']*test_set[name]
    return data_set,test_set

def combine_with_v28(list_of_columns,data_set,test_set):
    data_set=data_set.copy()
    test_set=test_set.copy()
    for name in list_of_columns:
        new_name='V28'+name
        data_set[new_name]=data_set['V28cutlabel']*data_set[name]
        test_set[new_name]=test_set['V28cutlabel']*test_set[name]
    return data_set,test_set

def combine_with_v35(list_of_columns,data_set,test_set):
    data_set=data_set.copy()
    test_set=test_set.copy()
    for name in list_of_columns:
        new_name='V35'+name
        data_set[new_name]=data_set['V35cutlabel']*data_set[name]
        test_set[new_name]=test_set['V35cutlabel']*test_set[name]
    return data_set,test_set


def combine_with_v9_v2(list_of_columns,data_set,test_set):
    data_set=data_set.copy()
    test_set=test_set.copy()
    accumulated_value_train=0
    accumulated_value_test=0
    for name in list_of_columns:
        new_name='V9cutlabel'+name
        accumulated_value_train=accumulated_value_train+data_set[name]
        accumulated_value_test=accumulated_value_test+test_set[name]
    data_set[new_name]=data_set['V9cutlabel']*accumulated_value_train
    test_set[new_name]=test_set['V9cutlabel']*accumulated_value_test
    return data_set,test_set



zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v9(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
# zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v17(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
# zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v22(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v23(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v24(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v28(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points,zhengqi_test_after_change_points=combine_with_v35(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)


# In[ ]:


"""
增加连续变量平方变化
"""

def add_squared_feature(column_names,train_set,test_set):
    train_set=train_set.copy()
    test_set=test_set.copy()
    for name in column_names:
        new_name='Squared'+name
        train_set[new_name]=train_set[name]*train_set[name]
        test_set[new_name]=test_set[name]*test_set[name]
    return train_set,test_set

zhengqi_train_after_drop_points,zhengqi_test_after_change_points=add_squared_feature(list_of_continous_columns,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
        


# In[ ]:


print(zhengqi_test_after_change_points.columns)
print(zhengqi_train_after_drop_points.columns)


# In[ ]:


"""
可选，drop columns去生成多个训练集，可选，经实践证明，不好用
"""

# drop features whose distributions are higly different

# 由于V5和V11高度线性相关，所以drop掉了分布非常不匹配的V5
list_of_columns1=['V5']
list_of_columns2=['V5','V22']
list_of_columns3=['V5','V11','V17','V22','V27']
list_of_columns4=['V5','V14','V21','V27','V32','V33']
list_of_columns5=['V5','V11','V14','V17','V22','V27']
# list_of_columnsx=['V5','V9','V9cut','V14','V22']
# list_of_columnsx=['V5','V14','V22']
# list_of_columnsx=['V4','V5','V11','V13','V19','V21','V22','V26','V28','V9','V17','V9cut','V17cut','V35','V35cut']
list_of_columnsx=['V4','V5','V11','V13','V19','V21','V22','V26','V28','V9cutlabel','V17cutlabel','V9cut','V17cut','V35cutlabel','V35cut','V17']
zhengqi_train_after_drop_points_try,zhengqi_test_try=drop_columns(list_of_columnsx,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)

def drop_columns(list_of_columns,train_data_frame,test_data_frame):
    zhengqi_train_after_drop=train_data_frame.drop(list_of_columns,axis=1)
    zhengqi_test_after_drop=test_data_frame.drop(list_of_columns,axis=1)
    return zhengqi_train_after_drop,zhengqi_test_after_drop

zhengqi_train_after_drop_points_without5,zhengqi_test_without5=drop_columns(list_of_columns1,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points_without522,zhengqi_test_without522=drop_columns(list_of_columns2,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points_without511172227,zhengqi_test_without511172227=drop_columns(list_of_columns3,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points_without51421273233,zhengqi_test_without51421273233=drop_columns(list_of_columns4,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
zhengqi_train_after_drop_points_without51114172227,zhengqi_test_without51114172227=drop_columns(list_of_columns5,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)

zhengqi_train_after_drop_points_try,zhengqi_test_try=drop_columns(list_of_columnsx,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)

print(zhengqi_train_after_drop_points_without5.columns,zhengqi_train_after_drop_points_without5.shape)
print(zhengqi_test_without5.columns,zhengqi_test_without5.shape)
print(zhengqi_train_after_drop_points_without522.columns,zhengqi_train_after_drop_points_without522.shape)
print(zhengqi_test_without522.columns,zhengqi_test_without522.shape)
print(zhengqi_train_after_drop_points_without511172227.columns,zhengqi_train_after_drop_points_without511172227.shape)
print(zhengqi_test_without511172227.columns,zhengqi_test_without511172227.shape)


# In[ ]:


def print_highly_correlated_pairs(data_frame,first_n):
    assert isinstance(data_frame,pd.DataFrame)
    
    print("找出最相关参数")
    corr = data_frame.corr()
    corr.sort_values(["target"], ascending = False, inplace = True)
    print(corr.target)
    
    print('other related feature pairs')
    data_frame_cor=data_frame.corr().abs().unstack().sort_values(kind="quicksort",ascending=False)
    data_frame_cor_filtered=data_frame_cor[data_frame_cor>0]
    data_frame_cor_filtered=data_frame_cor_filtered[data_frame_cor_filtered<1]
#     print(data_frame_cor_filtered)
    print(data_frame_cor_filtered[:first_n:2])
    
print_highly_correlated_pairs(zhengqi_train_after_drop_points,1000)


# In[ ]:


"""
可选，归一化或不归一化两种测试集，并把target单独拿出来
"""

"""
PCA降维处理
"""
# split features and target
# from sklearn.model_selection import train_test_split
# train_,test_=train_test_split(zhengqi_train_after_drop_points,test_size=0.2,random_state=2)
# test_x,test_y=split_feature_label(test_,'target')
# test_x_scaled=pd.DataFrame(scaler.fit_transform(test_x))

def scale_data(train_x,test_x):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    scaler=MinMaxScaler()
    scaler.fit(train_x)
    train_x_scaled=pd.DataFrame(scaler.transform(train_x))
    test_x_scaled=pd.DataFrame(scaler.transform(test_x))
    return train_x_scaled,test_x_scaled
    
def split_feature_label(dataset,column_name):
    feature_columns=dataset.drop(column_name,axis=1)
    return feature_columns,dataset.loc[:,column_name]

def pca_process(train_x,test_x):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    pca.fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)
    return train_x_pca,test_x_pca

train_x_without5,train_y_without5=split_feature_label(zhengqi_train_after_drop_points_without5,'target')
train_x_without522,train_y_without522=split_feature_label(zhengqi_train_after_drop_points_without522,'target')
train_x_without511172227,train_y_without511172227=split_feature_label(zhengqi_train_after_drop_points_without511172227,'target')
train_x_without51421273233,train_y_without51421273233=split_feature_label(zhengqi_train_after_drop_points_without51421273233,'target')
train_x_without51114172227,train_y_without51114172227=split_feature_label(zhengqi_train_after_drop_points_without51114172227,'target')
train_x_try,train_y_try=split_feature_label(zhengqi_train_after_drop_points_try,'target')


assert train_y_without5.shape[0]==zhengqi_train_after_drop_points_without5.shape[0]

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler_without5=MinMaxScaler()
scaler_without5.fit(train_x_without5)
train_x_without5_scaled=pd.DataFrame(scaler_without5.transform(train_x_without5))
zhengqi_test_without5_scaled=pd.DataFrame(scaler_without5.transform(zhengqi_test_without5))

scaler_without522=MinMaxScaler()
scaler_without522.fit(train_x_without522)
train_x_without522_scaled=pd.DataFrame(scaler_without522.transform(train_x_without522))
zhengqi_test_without522_scaled=pd.DataFrame(scaler_without522.transform(zhengqi_test_without522))

scaler_without511172227=MinMaxScaler()
scaler_without511172227.fit(train_x_without511172227)
train_x_without511172227_scaled=pd.DataFrame(scaler_without511172227.transform(train_x_without511172227))
zhengqi_test_without511172227_scaled=pd.DataFrame(scaler_without511172227.transform(zhengqi_test_without511172227))

train_x_without51114172227_scaled,zhengqi_test_without51114172227_scaled=scale_data(train_x_without51114172227,zhengqi_test_without51114172227)
print(train_x_try.columns)
train_x_try_scaled,zhengqi_test_try_scaled=scale_data(train_x_try,zhengqi_test_try)
print(train_x_try['V9cut'])
train_x_try_scaled_pca,zhengqi_test_try_scaled_pca=pca_process(train_x_try_scaled,zhengqi_test_try_scaled)

assert zhengqi_test_without511172227_scaled.shape[0]==zhengqi_test_after_change_points.shape[0]


# In[ ]:


def scale_data(train_x,test_x):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    scaler=MinMaxScaler()
    scaler.fit(train_x)
    train_x_scaled=pd.DataFrame(scaler.transform(train_x))
    test_x_scaled=pd.DataFrame(scaler.transform(test_x))
    return train_x_scaled,test_x_scaled
    
def split_feature_label(dataset,column_name):
    feature_columns=dataset.drop(column_name,axis=1)
    return feature_columns,dataset.loc[:,column_name]

def drop_columns(list_of_columns,train_data_frame,test_data_frame):
    zhengqi_train_after_drop=train_data_frame.drop(list_of_columns,axis=1)
    zhengqi_test_after_drop=test_data_frame.drop(list_of_columns,axis=1)
    return zhengqi_train_after_drop,zhengqi_test_after_drop

list_of_columnsx=['V5','V11','V27','V17','V22']
zhengqi_train_after_drop_points_try,zhengqi_test_try=drop_columns(list_of_columnsx,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
train_x_try,train_y_try=split_feature_label(zhengqi_train_after_drop_points_try,'target')
train_x_try_scaled,zhengqi_test_try_scaled=scale_data(train_x_try,zhengqi_test_try)


# In[ ]:


"""
第四步，创建index mask为最后的预测整合做准备
"""
# 1925*1,record indexes
record=np.zeros((zhengqi_test_after_change_points.shape[0],1))

# 用没有归一化，但是又处理了异常值的train和test set
for name in ['V27']:
    min_value_train=zhengqi_train_after_drop_points[name].min()
    max_value_train=zhengqi_train_after_drop_points[name].max()
    range_of_column=max_value_train-min_value_train
#     如果test中的一行在training set中对应的column的范围内，那么该index所在行就乘上对应的编号
#     if name=='V11':
#         record11=((zhengqi_test_after_change_points[name]<=max_value_train*0.7)&(zhengqi_test_after_change_points[name]>=min_value_train*0.7))*11
#     if name=='V17':
#         record17=((zhengqi_test_after_change_points[name]<=max_value_train*0.6)&(zhengqi_test_after_change_points[name]>=min_value_train*0.6))*17
#     if name=='V22':
#         record22=((zhengqi_test_after_change_points[name]<=max_value_train*0.7)&(zhengqi_test_after_change_points[name]>=min_value_train*0.7))*22
    if name=='V27':
        record27=((zhengqi_test_after_change_points[name]<=max_value_train*0.9)&(zhengqi_test_after_change_points[name]>=min_value_train*0.7))*27

record=record27


# In[ ]:


# result[index_of_points]=prediction1
assert record.shape[0]==1925

print(record.value_counts())

# index_of_points_using1727=record==44 #use without5 trainset
index_of_points_using27=record==27 #use without522 tarinset

# index_of_points_not_use1=record!=44
index_of_points_not_use=record!=27
# index_of_points_not_use=index_of_points_not_use1&index_of_points_not_use2 #use without511172227 train set

assert (np.sum(index_of_points_not_use)+np.sum(index_of_points_using27))==1925


# In[ ]:


"""
第五步， tune训练所用的模型

定义function for finding best parameters and training
"""

def find_best_parameter_for_kernel_ridge(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.kernel_ridge import KernelRidge
    LGBM_params = {'kernel': 'poly', 'alpha':1,'gamma':None,'degree':3,'coef0':1}

    back_params = {
#         'kernel':['rbf','poly'],
        "alpha": np.linspace(0.01,1,20),
        "gamma": np.logspace(-2, 2, 5),
        'degree':[2, 3, 4, 5,8,],
        'coef0':[0.5, 1, 1.5, 2],
    }
    for param in back_params:
        temp_param = {param: back_params[param]}
        estimator = KernelRidge(**LGBM_params)
        optimized_LGBM = GridSearchCV(estimator, param_grid=temp_param, scoring='neg_mean_squared_error',
                                      cv=5, verbose=False, n_jobs=4)
        optimized_LGBM.fit(train_x, train_y)

        LGBM_params.update(optimized_LGBM.best_params_)
        print('参数的最佳取值：{0}'.format(optimized_LGBM.best_params_))
        print('最佳模型得分:{0}'.format(-optimized_LGBM.best_score_))
    print(LGBM_params)


def find_best_parameter_for_lgbm(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from lightgbm import LGBMRegressor

    LGBM_params = {'num_leaves': 50, 'max_depth': 13, 'learning_rate': 0.1,
                   'n_estimators': 400, 'min_child_weight': 1, 'subsample': 0.8,
                   'colsample_bytree': 0.8, 'nthread': 4, 'objective': 'regression'}

    back_params = {
        'n_estimators': [i for i in range(400, 900, 20)],
        'num_leaves': [i for i in range(10, 45, 5)],
        'max_depth': [i for i in range(3, 11)],
        'min_child_weight': [i for i in range(1, 7)],
        'subsample': np.linspace(0.1, 0.9, 9),
        'colsample_bytree': np.linspace(0.1, 0.9, 9),
        'learning_rate': np.linspace(0.01, 0.2, 25),
    }
    for param in back_params:
        temp_param = {param: back_params[param]}
        estimator = LGBMRegressor(**LGBM_params)
        optimized_LGBM = GridSearchCV(estimator, param_grid=temp_param, scoring='neg_mean_squared_error',
                                      cv=5, verbose=False, n_jobs=4)
        optimized_LGBM.fit(train_x, train_y)

        LGBM_params.update(optimized_LGBM.best_params_)
        print('参数的最佳取值：{0}'.format(optimized_LGBM.best_params_))
        print('最佳模型得分:{0}'.format(-optimized_LGBM.best_score_))
    print(LGBM_params)


def find_best_parameter_for_xgboost(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb
    xgb_params = {'learning_rate': 0.1, 'n_estimators': 500,
                  'max_depth': 5, 'min_child_weight': 1,
                  'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    back_params = {
        'n_estimators': [i for i in range(500, 800, 10)],
        'max_depth': [i for i in range(3, 11)],
        'min_child_weight': [i for i in range(1, 7)],
        'gamma': np.linspace(0, 1, 11),
        'subsample': np.linspace(0.1, 0.9, 9),
        'colsample_bytree': np.linspace(0.1, 0.9, 9),
        'reg_alpha': np.linspace(0.1, 3, 30),
        'reg_lambda': np.linspace(0.1, 3, 30),
        'learning_rate': np.linspace(0.01, 0.2, 25),
    }
    for param in back_params:
        temp_param = {param: back_params[param]}
        estimator = xgb.XGBRegressor(**xgb_params)
        optimized_XGB = GridSearchCV(estimator, param_grid=temp_param, scoring='neg_mean_squared_error',
                                     cv=5, verbose=False, n_jobs=4)
        optimized_XGB.fit(train_x, train_y)

        xgb_params.update(optimized_XGB.best_params_)
        print('参数的最佳取值:{0}'.format(optimized_XGB.best_params_))
        print('最佳模型得分:{0}'.format(-optimized_XGB.best_score_))
    print(xgb_params)


def find_best_parameter_for_catboost(train_x, train_y):
    from catboost import CatBoostRegressor
    from sklearn.model_selection import GridSearchCV
    cat_params = {'n_estimators': 82,
                  'depth': 5,
                  'learning_rate': 0.1,
                  'l2_leaf_reg': 3,
                  'loss_function': 'RMSE',
                  'logging_level': 'Silent'}

    back_params = {
        'n_estimators': [i for i in range(400, 900, 25)],
        'depth': [i for i in range(1, 10, 1)],
        'learning_rate': np.linspace(0.01, 0.2, 20),
        'l2_leaf_reg': [i for i in range(1, 6, 1)],
    }
    for param in back_params:
        temp_param = {param: back_params[param]}
        estimator = CatBoostRegressor(**cat_params)
        optimized_CAT = GridSearchCV(estimator, param_grid=temp_param, scoring='neg_mean_squared_error',
                                     cv=5, verbose=False, n_jobs=4)
        optimized_CAT.fit(train_x, train_y)

        cat_params.update(optimized_CAT.best_params_)
        print('参数的最佳取值：{0}'.format(optimized_CAT.best_params_))
        print('最佳模型得分:{0}'.format(-optimized_CAT.best_score_))
    print(cat_params)
    
def find_best_para_for_gbr(train_x,train_y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GridSearchCV
    print("search for gbr******")
    gbr_params = {'learning_rate':0.03, 'loss':'huber', 'max_depth':3,
              'min_impurity_decrease':0.0, 'min_samples_leaf':1, 'min_samples_split':2,
              'n_estimators':100, 'random_state':0, 'subsample':0.8}
    back_params = {
        'max_depth': [i for i in range(5,15,1)],
        'n_estimators': [i for i in range(75,500,25)],
        'learning_rate':np.linspace(0.01,0.1,10),
        'subsample': np.linspace(0.01,0.1,10),
        'min_samples_leaf': [i for i in range(1,15,1)],
        'min_samples_split': [i for i in range(2,42,2)]
    }
    for param in back_params:
        temp_param = {param:back_params[param]}
        estimator = GradientBoostingRegressor(**gbr_params)
        optimized_gbr = GridSearchCV(estimator, param_grid = temp_param, 
                                     scoring='neg_mean_squared_error',
                                     cv=5, verbose=False, n_jobs=4)
        optimized_gbr.fit(train_x, train_y)

        gbr_params.update(optimized_gbr.best_params_)
        print('参数的最佳取值：{0}'.format(optimized_gbr.best_params_))
        print('最佳模型得分:{0}'.format(-optimized_gbr.best_score_))
    print(gbr_params)
           

def find_best_para_for_rf(train_x,train_y):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    print("search for random forest*****")
    gbr_params = { 'max_depth':3,
               'min_samples_leaf':1, 'min_samples_split':2,
              'n_estimators':100, 'random_state':0}
    back_params = {
        'max_depth': [i for i in range(5,15,1)],
        'n_estimators': [i for i in range(75,500,25)],
        'min_samples_leaf': [i for i in range(1,15,1)],
        'min_samples_split': [i for i in range(2,42,2)],
        "max_leaf_nodes": [i for i in range(10,100,10)],
        "min_weight_fraction_leaf": np.linspace(0.05,0.3,10)
    }
    for param in back_params:
        temp_param = {param:back_params[param]}
        estimator = RandomForestRegressor(**gbr_params)
        optimized_gbr = GridSearchCV(estimator, param_grid = temp_param, 
                                     scoring='neg_mean_squared_error',
                                     cv=5, verbose=False, n_jobs=4)
        optimized_gbr.fit(train_x, train_y)

        gbr_params.update(optimized_gbr.best_params_)
        print('参数的最佳取值：{0}'.format(optimized_gbr.best_params_))
        print('最佳模型得分:{0}'.format(-optimized_gbr.best_score_))
    print(gbr_params)

def train_xgb_and_predict(parameter, train_x, train_y, test_x):
    import xgboost as xgb
    from xgboost import plot_importance
    from sklearn.metrics import mean_squared_error
    xgb_model = xgb.XGBRegressor(**parameter)
    xgb_model.fit(train_x, train_y)
    train_pred = xgb_model.predict(train_x)
    test_pred = xgb_model.predict(test_x)
    print(mean_squared_error(train_y,train_pred))
    my_xgb_plot_importance(xgb_model,(16,8))
    return train_pred, test_pred


def train_lgbm_and_predict(parameter, train_x, train_y, test_x):
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_error
    lgbm_model = LGBMRegressor(**parameter)
    lgbm_model.fit(train_x, train_y)
    train_pred = lgbm_model.predict(train_x)
    test_pred = lgbm_model.predict(test_x)
    print(mean_squared_error(train_y,train_pred))
    my_lgbm_plot_importance(lgbm_model,(16,8))
    return train_pred, test_pred


def train_catboost_and_predict(parameter, train_x, train_y, test_x):
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    cat_model = CatBoostRegressor(**parameter)
    cat_model.fit(train_x, train_y)
    train_pred = cat_model.predict(train_x)
    test_pred = cat_model.predict(test_x)
    print(mean_squared_error(train_y,train_pred))
#     my_cat_plot_importance(cat_model,(16,8))
    return train_pred, test_pred

def my_xgb_plot_importance(booster, figsize, **kwargs): 
    from matplotlib import pyplot as plt
    from xgboost import plot_importance
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, **kwargs)


def my_lgbm_plot_importance(booster, figsize, **kwargs): 
    from matplotlib import pyplot as plt
    from lightgbm import plot_importance
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax, **kwargs)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def kfold_scores(alg,x_train,y_train,is_nn=False):
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    import tensorflow as tf
    kf = KFold(n_splits = 5, random_state= 1, shuffle=False)
    predict_y = []
    
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=1,
        )
    ]
    if is_nn:
        alg.save_weights('initial_weights.h5')
        
    for kf_train,kf_test in kf.split(x_train):
        if is_nn:
            alg.load_weights('initial_weights.h5')
            alg.fit(x_train.iloc[kf_train],y_train.iloc[kf_train],epochs=10,callbacks=callbacks_list,validation_data=(x_train.iloc[kf_test],y_train.iloc[kf_test]),verbose=2)
        else:
            alg.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
        y_pred_train = alg.predict(x_train.iloc[kf_test])
        mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
        predict_y.append(mse)
    
    cv_mse=np.mean(predict_y)
    print("交叉验证集MSE均值为 %s" % (np.mean(predict_y)))  
    return cv_mse

def kfold_scores_v2(name_of_model,para,x_train,y_train):
    kf = KFold(n_splits = 5, random_state= 1, shuffle=False)

    predict_y = []
    for kf_train,kf_test in kf.split(x_train):
        if name_of_model=='xgb':
            import xgboost as xgb
            from sklearn.metrics import mean_squared_error
            alg = xgb.XGBRegressor(**para)
        if name_of_model=='lgbm':
            from lightgbm import LGBMRegressor
            from sklearn.metrics import mean_squared_error
            alg = LGBMRegressor(**para)
        if name_of_model=='cat':
            from catboost import CatBoostRegressor
            from sklearn.metrics import mean_squared_error
            alg = CatBoostRegressor(**para)
            
        alg.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
        y_pred_train = alg.predict(x_train.iloc[kf_test])
        mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
        predict_y.append(mse)
    
    cv_mse=np.mean(predict_y)
    print("交叉验证集MSE均值为 %s" % (np.mean(predict_y)))
    
    return cv_mse

def kfold_scores_v3(para_xgb,para_lgb,x_train,y_train,l=1,x=1):
    kf = KFold(n_splits = 5, random_state= 1, shuffle=False)

    predict_y = []
    for kf_train,kf_test in kf.split(x_train):

        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        alg1 = xgb.XGBRegressor(**para_xgb)

        from lightgbm import LGBMRegressor
        alg2 = LGBMRegressor(**para_lgb)
            
        alg1.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
        alg2.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])

        y_pred_train1 = alg1.predict(x_train.iloc[kf_test])
        y_pred_train2 = alg2.predict(x_train.iloc[kf_test])
        y_pred_train=(y_pred_train1*x+y_pred_train2*l)/(l+x)
        mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
        predict_y.append(mse)
    
    cv_mse=np.mean(predict_y)
    print("交叉验证集MSE均值为 %s" % (np.mean(predict_y)))
    return cv_mse

def kfold_scores_v4(para_xgb,para_lgb,alpha_ridge,x_train,y_train):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    kf = KFold(n_splits = 5, random_state= 1, shuffle=False)
    best = [0,0,0,0,10]
    min_mse=10
    for x in np.linspace(0.1,1,5):
        for l in np.linspace(0.1,1,5):
            for la in np.linspace(0.1,1,5):
                predict_y = []
                for kf_train,kf_test in kf.split(x_train):
                    alg1 = xgb.XGBRegressor(**para_xgb)
                    alg2 = LGBMRegressor(**para_lgb)
                    alg3=Ridge(alpha_ridge)

                    alg1.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
                    alg2.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
                    alg3.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])


                    y_pred_train1 = alg1.predict(x_train.iloc[kf_test])
                    y_pred_train2 = alg2.predict(x_train.iloc[kf_test])
                    y_pred_train3 = alg3.predict(x_train.iloc[kf_test])

                    y_pred_train=(y_pred_train1*x+y_pred_train2*l+y_pred_train3*la)/(la+l+x)
                    mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
                    predict_y.append(mse)
                cv_mse=np.mean(predict_y)
                if cv_mse<min_mse:
                    min_mse=cv_mse
                    best=[x,l,la,min_mse]
    print("交叉验证集MSE最佳均值为 %s" % (best[3]))
    print("params are:",best)
    return cv_mse

def train_ridge(column_names,train_x=zhengqi_train_after_drop_points,test_x=zhengqi_test_after_change_points):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    zhengqi_train_after_drop_points_try,zhengqi_test_try=drop_columns(column_names,train_x,test_x)
    train_x_try,train_y_try=split_feature_label(zhengqi_train_after_drop_points_try,'target')
    train_x_try_scaled,zhengqi_test_try_scaled=scale_data(train_x_try,zhengqi_test_try)
    ridge = Ridge(random_state=len(column_names))
    params = {
            'alpha':np.linspace(0.01, 1, 100),
             }
    grid = GridSearchCV(estimator=ridge, 
                        param_grid=params, 
                        scoring='neg_mean_squared_error',
                        cv=5, verbose=False)
    kfold_scores(grid,train_x_try_scaled,train_y_try)
    grid.fit(train_x_try_scaled,train_y_try)
    prediction=grid.predict(zhengqi_test_try_scaled)
    print("best mse during training:"+str(grid.best_score_))
    print("final whole data mse:",mean_squared_error(grid.predict(train_x_try_scaled),train_y_try))
    print("best parameter:",grid.best_params_)
    print("training set columns:",train_x_try.columns)
    return grid,prediction


# In[ ]:


"""
Lasso feature selection
"""

def ridge_feature_selection(train_x,train_y,names=train_x_try.columns):
    from sklearn.linear_model import Ridge
    print("starting feature number is:",len(train_x_try.columns))
    train_x=train_x.copy()
    train_x.columns=names
    record_mse_for_each_round=[]
    record_dropname_for_each_round=[]
    global_min_mse=1
    global_min_mse_round=-1
    rounds=0
    while len(train_x.columns)>20:
        min_mse=1
        rounds=rounds+1
        for name in train_x.columns:
            print("round{}:,drop {}*********".format(rounds,name))
            ridge=Ridge(random_state=len(train_x.columns))
            train_x_drop=train_x.drop([name],axis=1)
            cv_mse=kfold_scores(ridge,train_x_drop,train_y)
            if cv_mse<min_mse:
                min_mse=cv_mse
                drop_name=name
        record_mse_for_each_round.append(min_mse)
        record_dropname_for_each_round.append(drop_name)
        train_x=train_x.drop([drop_name],axis=1)
        if min_mse<global_min_mse:
            global_min_mse=min_mse
            global_min_mse_round=rounds
            columns_at_this_round=train_x.columns
    print("best mse is {} at rounds {}".format(global_min_mse,global_min_mse_round))
    print("columns name at this round are:",columns_at_this_round,"***number is",str(len(columns_at_this_round)))
    plt.figure(figsize=(30,16))
    plt.xlabel("Number of features dropped")
    plt.ylabel("Cross validation score")
    plt.plot(range(1,rounds+1), record_mse_for_each_round)
    plt.scatter(range(1,rounds+1), record_mse_for_each_round,marker='x',color='red')
    plt.xticks(np.arange(1, rounds+1, 1.0))
    plt.show()
    return record_mse_for_each_round,record_dropname_for_each_round,global_min_mse,rounds

"""
Lasso feature selection
"""

def svr_feature_selection(train_x,train_y,names=train_x_try.columns):
    from sklearn import svm
    print("starting feature number is:",len(train_x_try.columns))
    train_x=train_x.copy()
    train_x.columns=names
    record_mse_for_each_round=[]
    record_dropname_for_each_round=[]
    global_min_mse=1
    global_min_mse_round=-1
    rounds=0
    while len(train_x.columns)>20:
        min_mse=1
        rounds=rounds+1
        for name in train_x.columns:
            print("round{}:,drop {}*********".format(rounds,name))
            ridge=svm.SVR()
            train_x_drop=train_x.drop([name],axis=1)
            cv_mse=kfold_scores(ridge,train_x_drop,train_y)
            if cv_mse<min_mse:
                min_mse=cv_mse
                drop_name=name
        record_mse_for_each_round.append(min_mse)
        record_dropname_for_each_round.append(drop_name)
        train_x=train_x.drop([drop_name],axis=1)
        if min_mse<global_min_mse:
            global_min_mse=min_mse
            global_min_mse_round=rounds
            columns_at_this_round=train_x.columns
    print("best mse is {} at rounds {}".format(global_min_mse,global_min_mse_round))
    print("columns name at this round are:",columns_at_this_round,"***number is",str(len(columns_at_this_round)))
    plt.figure(figsize=(30,16))
    plt.xlabel("Number of features dropped")
    plt.ylabel("Cross validation score")
    plt.plot(range(1,rounds+1), record_mse_for_each_round)
    plt.scatter(range(1,rounds+1), record_mse_for_each_round,marker='x',color='red')
    plt.xticks(np.arange(1, rounds+1, 1.0))
    plt.show()
    return record_mse_for_each_round,record_dropname_for_each_round,global_min_mse,rounds

def bridge_feature_selection(train_x,train_y,names=train_x_try.columns):
    from sklearn.linear_model import BayesianRidge
    print("starting feature number is:",len(train_x_try.columns))
    train_x=train_x.copy()
    train_x.columns=names
    record_mse_for_each_round=[]
    record_dropname_for_each_round=[]
    global_min_mse=1
    global_min_mse_round=-1
    rounds=0
    while len(train_x.columns)>20:
        min_mse=1
        rounds=rounds+1
        for name in train_x.columns:
            print("round{}:,drop {}*********".format(rounds,name))
            ridge=BayesianRidge()
            train_x_drop=train_x.drop([name],axis=1)
            cv_mse=kfold_scores(ridge,train_x_drop,train_y)
            if cv_mse<min_mse:
                min_mse=cv_mse
                drop_name=name
        record_mse_for_each_round.append(min_mse)
        record_dropname_for_each_round.append(drop_name)
        train_x=train_x.drop([drop_name],axis=1)
        if min_mse<global_min_mse:
            global_min_mse=min_mse
            global_min_mse_round=rounds
            columns_at_this_round=train_x.columns
    print("best mse is {} at rounds {}".format(global_min_mse,global_min_mse_round))
    print("columns name at this round are:",columns_at_this_round,"***number is",str(len(columns_at_this_round)))
    plt.figure(figsize=(30,16))
    plt.xlabel("Number of features dropped")
    plt.ylabel("Cross validation score")
    plt.plot(range(1,rounds+1), record_mse_for_each_round)
    plt.scatter(range(1,rounds+1), record_mse_for_each_round,marker='x',color='red')
    plt.xticks(np.arange(1, rounds+1, 1.0))
    plt.show()
    return record_mse_for_each_round,record_dropname_for_each_round,global_min_mse,rounds


# In[ ]:


"""
第一次尝试的方案，通过训练集的最大最小值把测试集分成两个部分，符合训练集的部分以及超过训练集的部分，
然后对这两个测试集分别进行预测，最后通过numpy mask整合在一起效果不好
"""


# In[ ]:


np.count_nonzero(final_prediction_xgb_lgbm_ridge_scaled_feature_transformed_5_and_527)


# In[ ]:


"""
1
归一化的版本，lgmb，使用了11172227这些column
"""

find_best_parameter_for_lgbm(train_x_without5_scaled,train_y_without5)


# In[ ]:


para={'num_leaves': 20, 'max_depth': 3, 'learning_rate': 0.04, 'n_estimators': 400, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.8, 'nthread': 4, 'objective': 'regression'}
train_pred21,test_pred21=train_lgbm_and_predict(para,train_x_without5_scaled,train_y_without5,zhengqi_test_without5_scaled)
kfold_scores_v2('lgbm',para,train_x_without5_scaled,train_y_without5)


# In[ ]:


"""
2
归一化的版本，lgmb，使用了111727这些column
"""

find_best_parameter_for_lgbm(train_x_without522_scaled,train_y_without522)


# In[ ]:


para={'num_leaves': 40, 'max_depth': 9, 'learning_rate': 0.02, 'n_estimators': 850, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.8, 'nthread': 4, 'objective': 'regression'}
train_pred22,test_pred22=train_lgbm_and_predict(para,train_x_without522_scaled,train_y_without522,zhengqi_test_without522_scaled)
kfold_scores_v2('lgbm',para,train_x_without522_scaled,train_y_without522)


# In[ ]:


"""
3
归一化的版本，lgbm,5111727这些column全都没使用
"""
find_best_parameter_for_lgbm(train_x_without511172227_scaled,train_y_without511172227)


# In[ ]:


para={'num_leaves': 40, 'max_depth': 9, 'learning_rate': 0.03, 'n_estimators': 425, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.7000000000000001, 'nthread': 4, 'objective': 'regression'}
train_pred23,test_pred23=train_lgbm_and_predict(para,train_x_without511172227_scaled,train_y_without511172227,zhengqi_test_without511172227_scaled)
kfold_scores_v2('lgbm',para,train_x_without511172227_scaled,train_y_without511172227)


# In[ ]:


# store the final result for lgbm with scaled
final_prediction_lgbm_scaled=np.zeros((zhengqi_test.shape[0],1))

final_prediction_lgbm_scaled[index_of_points_using11172227]=test_pred21[index_of_points_using11172227].reshape(-1,1)
final_prediction_lgbm_scaled[index_of_points_using111727]=test_pred22[index_of_points_using111727].reshape(-1,1)
final_prediction_lgbm_scaled[index_of_points_not_use]=test_pred23[index_of_points_not_use].reshape(-1,1)


# In[ ]:


"""
1
归一化的版本，xgboost，使用了11172227这些column
"""

find_best_parameter_for_xgboost(train_x_without5_scaled,train_y_without5)


# In[ ]:


para={'learning_rate': 0.06, 'n_estimators': 510, 'max_depth': 5, 'min_child_weight': 2, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.0, 'reg_alpha': 0.6, 'reg_lambda': 1.7}
train_pred41,test_pred41=train_xgb_and_predict(para,train_x_without5_scaled,train_y_without5,zhengqi_test_without5_scaled)
kfold_scores_v2('xgb',para,train_x_without5_scaled,train_y_without5)


# In[ ]:


"""
2
归一化的版本，xgboost，使用了111727这些column
"""
find_best_parameter_for_xgboost(train_x_without522_scaled,train_y_without522)


# In[ ]:


para={'learning_rate': 0.1, 'n_estimators': 400, 'max_depth': 3, 'min_child_weight': 6, 'seed': 0, 'subsample': 0.90000000000000002, 'colsample_bytree': 0.80000000000000004, 'gamma': 0.80000000000000004, 'reg_alpha': 1.3, 'reg_lambda': 1.8}
train_pred42,test_pred42=train_xgb_and_predict(para,train_x_without522_scaled,train_y_without522,zhengqi_test_without522_scaled)
kfold_scores_v2('xgb',para,train_x_without522_scaled,train_y_without522)


# In[ ]:


"""
3
归一化的版本，xgboost,5111727这些column全都没使用
"""
find_best_parameter_for_xgboost(train_x_without511172227_scaled,train_y_without511172227)


# In[ ]:


para={'learning_rate': 0.1, 'n_estimators': 450, 'max_depth': 3, 'min_child_weight': 5, 'seed': 0, 'subsample': 0.90000000000000002, 'colsample_bytree': 0.59999999999999998, 'gamma': 0.70000000000000007, 'reg_alpha': 0.20000000000000001, 'reg_lambda': 1.0999999999999999}
train_pred43,test_pred43=train_xgb_and_predict(para,train_x_without511172227_scaled,train_y_without511172227,zhengqi_test_without511172227_scaled)
kfold_scores_v2('xgb',para,train_x_without511172227_scaled,train_y_without511172227)


# In[ ]:


# store the final result for xgboost with scaled
final_prediction_xgb_scaled=np.zeros((zhengqi_test.shape[0],1))

final_prediction_xgb_scaled[index_of_points_using11172227]=test_pred41[index_of_points_using11172227].reshape(-1,1)
final_prediction_xgb_scaled[index_of_points_using111727]=test_pred42[index_of_points_using111727].reshape(-1,1)
final_prediction_xgb_scaled[index_of_points_not_use]=test_pred43[index_of_points_not_use].reshape(-1,1)


# In[ ]:


"""
最后一步，整合所有结果，并存储
"""
"""
scaled result
"""
result_xgb_lgbm_scaled_average=(final_prediction_xgb_scaled+final_prediction_lgbm_scaled)/2.0
np.savetxt('result_xgb_lgbm_scaled_average.txt',result_xgb_lgbm_scaled_average)

print(result_xgb_lgbm_scaled_average.shape)


# In[ ]:


"""
第二种尝试方案，训练多个模型，然后用库自带的stacking方法来集成或神经网络集成，效果也不好
"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,LinearRegression,Ridge
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingRegressor
from lightgbm import LGBMRegressor

para_xgb={'learning_rate': 0.07333333333333333, 'n_estimators': 500, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.4, 'gamma': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
para_lgb={'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.025833333333333333, 'n_estimators': 400, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.4, 'nthread': 4, 'objective': 'regression'}
alpha_ridge=0.06

alg1 = xgb.XGBRegressor(**para_xgb)
alg2 = LGBMRegressor(**para_lgb)
alg3=Ridge(alpha_ridge)


####1SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####4GBRT回归####
from sklearn import ensemble
para_gbr={'learning_rate': 0.020000000000000004, 'loss': 'huber', 'max_depth': 5, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 300, 'random_state': 0, 'subsample': 0.09000000000000001}
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(**para_gbr)
####5BayesianRidge贝叶斯岭回归
from sklearn.linear_model import BayesianRidge,TheilSenRegressor
model_BayesianRidge = BayesianRidge()
####6TheilSen泰尔森估算
model_TheilSenRegressor = TheilSenRegressor(n_jobs=2)
# 7
from sklearn.kernel_ridge import KernelRidge
model_KernelRidge=KernelRidge(alpha=0.06, kernel='polynomial', degree=3, coef0=1)
regressors = [alg1,alg2,alg3,model_KernelRidge,model_SVR,model_TheilSenRegressor,model_BayesianRidge,model_GradientBoostingRegressor]


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

gbr_params = {'learning_rate':0.03, 'loss':'huber', 'max_depth':3,
          'min_impurity_decrease':0.0, 'min_samples_leaf':1, 'min_samples_split':2,
          'n_estimators':100, 'random_state':0, 'subsample':0.8}
back_params = {
    'meta-gradientboostingregressor__max_depth': [i for i in range(5,15,1)],
    'meta-gradientboostingregressor__n_estimators': [i for i in range(75,500,25)],
    'meta-gradientboostingregressor__learning_rate':np.linspace(0.01,0.1,10),
    'meta-gradientboostingregressor__subsample': np.linspace(0.01,0.1,10),
    'meta-gradientboostingregressor__min_samples_leaf': [i for i in range(1,15,2)],
    'meta-gradientboostingregressor__min_samples_split': [i for i in range(2,38,4)]
}

meta_gbr = GradientBoostingRegressor(**gbr_params)

regressors = [alg1,alg2,alg3,model_KernelRidge,model_SVR,model_TheilSenRegressor,model_BayesianRidge,model_GradientBoostingRegressor]

stregr = StackingRegressor(regressors=regressors, 
                           meta_regressor=meta_gbr)

grid = GridSearchCV(estimator=stregr, 
                    param_grid=back_params, 
                    scoring='neg_mean_squared_error',
                    cv=5, verbose=False, n_jobs=4)
grid.fit(train_x_try_scaled, train_y_try)

print("best score",grid.best_score_)
print("best estimator",grid.best_estimator_)
print("best params",grid.best_params_)

# lr = LinearRegression()
# svr_lin = SVR(kernel='linear')
# ridge = Ridge(random_state=1)
# svr_rbf = SVR(kernel='rbf')

# stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], 
#                            meta_regressor=svr_rbf)

# stregr.fit(X, y)
# stregr.predict(X)


# In[ ]:


result=model_nn.predict(zhengqi_test_after_drop_columns_scaled)
np.savetxt('result.txt',result)

k = result.tolist()

with open('data.txt','w') as f:
    for i in k:
        f.write(str(i) + '\n')
    f.close()


# In[ ]:


# with scale
from joblib import dump, load
from sklearn.metrics import mean_squared_error
# xgb_params_mine={'learning_rate': 0.07, 'n_estimators': 570, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.0, 'reg_alpha': 0.3, 'reg_lambda': 1.5}
xgb_params_mine={'learning_rate': 0.02, 'n_estimators': 500, 'max_depth': 6, 'min_child_weight': 4, 'seed': 0, 'subsample': 0.20000000000000001, 'colsample_bytree': 0.70000000000000007, 'gamma': 0.0, 'reg_alpha': 0.20000000000000001, 'reg_lambda': 3.0}
xgb_model = xgb.XGBRegressor(**xgb_params_mine)
xgb_model.fit(train_x_scaled, train_y)
train_pred_xgb = xgb_model.predict(train_x_scaled)
dump(xgb_model,'xgb_model_scaled_without_511172227.joblb')

# lgb_params_self={'num_leaves': 20, 'max_depth': 7, 'learning_rate': 0.03, 'n_estimators': 875, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.4, 'nthread': 4, 'objective': 'regression'}
# {'num_leaves': 40, 'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 625, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.7000000000000001, 'nthread': 4, 'objective': 'regression'}
lgb_params_self={'num_leaves': 40, 'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 625, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.7000000000000001, 'nthread': 4, 'objective': 'regression'}
lgbm_model = LGBMRegressor(**lgb_params_self)
lgbm_model.fit(train_x_scaled, train_y)
train_pred_lgb = lgbm_model.predict(train_x_scaled)
dump(lgbm_model,'lgb_model_scaled_without_511172227.joblb')
train_pred=(train_pred_xgb+train_pred_lgb)/2.0
mean_squared_error(train_y,train_pred)


# In[ ]:


from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
def build_model(input_shape,num_of_layers,num_of_units,dropout_rate):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    assert num_of_layers>2
    model = models.Sequential()
    model.add(layers.Dense(num_of_units, activation='relu',
                               input_shape=(input_shape,)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.BatchNormalization())
#     add middle layers
    for i in range(num_of_layers-2):
        model.add(layers.Dense(num_of_units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.BatchNormalization())
        
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


# model.fit(train_x_scaled, train_y,
#               epochs=5, batch_size=64, verbose=0)
# print(model.metrics_names)
# model.evaluate(test_x_scaled,test_y)

min_cv_mse=10
min_test_mse=10
min_cv_mse_parameters=[]
min_test_mse_parameters=[]
for num_of_layers in [3,4,5]:
    for num_of_units in [128,256,512]:
        for drop_out_rate in [0,0.1]:
            print('num of layers is {}, num of units is {}, dropout rate is {}'.format(num_of_layers,num_of_units,drop_out_rate))
            cv_mse_list=[]
            test_mse_list=[]
            model=build_model(results_of_models.shape[1],num_of_layers,num_of_units,drop_out_rate)
            cv_mse=kfold_scores(model,results_of_models,train_y_try,True)
            cv_mse_list.append(cv_mse)
            if np.mean(cv_mse_list)<min_cv_mse:
                min_cv_mse=np.mean(cv_mse_list)
                min_cv_mse_parameters=[min_cv_mse,num_of_layers,num_of_units,drop_out_rate]
#             if np.mean(test_mse)<min_test_mse:
#                 min_test_mse=np.mean(test_mse_list)
#                 min_test_mse_parameters=[min_test_mse,num_of_layers,num_of_units,drop_out_rate]

print(min_cv_mse)
print(min_cv_mse_parameters)


# In[ ]:


"""
可选，没用到，遍历所有可能的组合并生成相对应的训练和测试集

"""

def drop_columns(list_of_columns,train_data_frame,test_data_frame):
    zhengqi_train_after_drop=train_data_frame.drop(list_of_columns,axis=1)
    zhengqi_test_after_drop=test_data_frame.drop(list_of_columns,axis=1)
    return zhengqi_train_after_drop,zhengqi_test_after_drop

# remove V5
zhengqi_train_after_drop_points,_=drop_columns(['V5'],zhengqi_train_after_drop_points,zhengqi_test)

from itertools import combinations
columns=['V11','V17','V22','V27']
all_combinations=[]
all_combination_data={}
for i in range(len(columns)):
    all_combinations.append(list(combinations(columns,i+1)))

for i in range(len(all_combinations)):
    for j in range(len(all_combinations[i])):
        key_train='train_without_'
        key_test='test_without_'
        record=''
        for name in all_combinations[i][j]:
            record=record+name
        key_train=key_train+record
        key_test=key_test+record
        all_combination_data[key_train],all_combination_data[key_test]=drop_columns(list(all_combinations[i][j]),zhengqi_train_after_drop_points,zhengqi_test)
        


# In[ ]:


"""
第三种方案，先去掉一些分布严重不一致的特征，然后分箱（分或者不分），然后组合，
然后用ridge或者bayesian ridge通过backward feature selection来筛选特征
这种方案根据是否分箱，以及如何组合特征，有多个可能性
"""


# In[ ]:


record_mse_for_each_round,record_dropname_for_each_round,global_min_mse,rounds=ridge_feature_selection(train_x_try_scaled,train_y_try)


# In[ ]:


record_mse_for_each_round,record_dropname_for_each_round,global_min_mse,rounds=bridge_feature_selection(train_x_try_scaled,train_y_try)


# In[ ]:


"""
根据上面两个的结果，选择特征
"""
dropped_columns_2=['V5','V17','V22','V27','V35V14', 'SquaredV15', 'V28V19', 'V23V14', 'V9V15', 'V24V1', 'V24V16', 'V24V29', 'V11', 'V23V16', 'V18', 'SquaredV3', 'V20', 'V24V10', 'V35V10', 'V23V29', 'SquaredV19', 'V35V15', 'V21', 'V23', 'V9V19', 'SquaredV0', 'V28V3', 'V24V36', 'V28V16', 'V24V0', 'V35V30', 'V9V30', 'V28V0', 'V9V29', 'V35V3', 'V28V30', 'V24V3', 'V35V36', 'V19', 'V9V10', 'SquaredV29', 'V9V3', 'V9V37', 'V35V19', 'V28V36', 'V25', 'V23V37', 'V24V14', 'SquaredV14', 'V24V37', 'V28', 'V28V37', 'V24V30', 'V30', 'V9V1', 'V23V19', 'V23V1', 'V24', 'SquaredV30', 'V35V29', 'V9V14', 'V29', 'V24V19', 'V34', 'V23V15', 'V28V10', 'V28V14', 'V26', 'V32', 'V6', 'V23V30', 'V35', 'V15', 'V9V0', 'V28V1', 'V31', 'V35V1', 'V35V37']
zhengqi_train_after_drop_points_try_2,zhengqi_test_try_2=drop_columns(dropped_columns_2,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
train_x_try_2,train_y_try_2=split_feature_label(zhengqi_train_after_drop_points_try_2,'target')
train_x_try_scaled_2,zhengqi_test_try_scaled_2=scale_data(train_x_try_2,zhengqi_test_try_2)
print(zhengqi_test_try_2.columns)


# In[ ]:


"""
看看用了哪些特征，哪些没用
"""
name_set={"V"}
for name in zhengqi_test_try_2.columns:
    for subname in zhengqi_test.columns:
        if subname in name:
            name_set.add(subname)
            
name_l=list(name_set)
print(name_l)
print(set(zhengqi_test.columns)-name_set)


# In[ ]:


# 看看筛选后的特征，训练集和测试集是否分布一致
plot_each_column(zhengqi_train_after_drop_points_try_2,zhengqi_test_try_2)


# In[ ]:


# tune模型参数
dropped_columns_2=['V5','V17','V22','V27','V35V14', 'SquaredV15', 'V28V19', 'V23V14', 'V9V15', 'V24V1', 'V24V16', 'V24V29', 'V11', 'V23V16', 'V18', 'SquaredV3', 'V20', 'V24V10', 'V35V10', 'V23V29', 'SquaredV19', 'V35V15', 'V21', 'V23', 'V9V19', 'SquaredV0', 'V28V3', 'V24V36', 'V28V16', 'V24V0', 'V35V30', 'V9V30', 'V28V0', 'V9V29', 'V35V3', 'V28V30', 'V24V3', 'V35V36', 'V19', 'V9V10', 'SquaredV29', 'V9V3', 'V9V37', 'V35V19', 'V28V36', 'V25', 'V23V37', 'V24V14', 'SquaredV14', 'V24V37', 'V28', 'V28V37', 'V24V30', 'V30', 'V9V1', 'V23V19', 'V23V1', 'V24', 'SquaredV30', 'V35V29', 'V9V14', 'V29', 'V24V19', 'V34', 'V23V15', 'V28V10', 'V28V14', 'V26', 'V32', 'V6', 'V23V30', 'V35', 'V15', 'V9V0', 'V28V1', 'V31', 'V35V1', 'V35V37']
train_ridge(dropped_columns_2,zhengqi_train_after_drop_points,zhengqi_test_after_change_points)
find_best_parameter_for_kernel_ridge(train_x_try_scaled_2,train_y_try_2)
print(train_x_try_2.columns)
find_best_parameter_for_lgbm(train_x_try_scaled_2,train_y_try)
find_best_parameter_for_xgboost(train_x_try_scaled_2,train_y_try)
find_best_para_for_gbr(train_x_try_scaled_2,train_y_try)


# In[ ]:


"""
只用两个，效果不好
"""


# In[ ]:


"""
bayesian ridge and ridge
"""
def kfold_scores_v6(x_train,y_train):
    kf = KFold(n_splits = 5, random_state= 1, shuffle=False)
    best = [0,0,10]
    min_mse=10
    i=0
    for x1 in np.linspace(0.1,1,10):
        for x2 in np.linspace(0.1,1,10):
#             for x3 in np.linspace(0.1,1,10):
#             for x4 in np.linspace(0.1,1,5):
            predict_y = []
            for kf_train,kf_test in kf.split(x_train):
                alg1 = BayesianRidge()
                alg2=Ridge(0.51)
#                             alg3 = BayesianRidge()
#                         alg3=KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
#                     alg3 = TheilSenRegressor()
#                     para_xgb={'learning_rate': 0.025833333333333333, 'n_estimators': 500, 'max_depth': 3, 'min_child_weight': 2, 'seed': 0, 'subsample': 0.6, 'colsample_bytree': 0.2, 'gamma': 0.9, 'reg_alpha': 0.2, 'reg_lambda': 1.2}
#                     alg3 = xgb.XGBRegressor(**para_xgb)

                alg1.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
                alg2.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
#                     alg3.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
#                         alg4.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
#                             alg5.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])


                y_pred_train1 = alg1.predict(x_train.iloc[kf_test])
                y_pred_train2 = alg2.predict(x_train.iloc[kf_test])
#                     y_pred_train3 = alg3.predict(x_train.iloc[kf_test])
#                         y_pred_train4 = alg4.predict(x_train.iloc[kf_test])
#                             y_pred_train5 = alg5.predict(x_train.iloc[kf_test])


                y_pred_train=(y_pred_train1*x1+y_pred_train2*x2)/(x1+x2)
                mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
                predict_y.append(mse)
            cv_mse=np.mean(predict_y)
            print("current turn mse:",cv_mse,"turn is",i)
            i=i+1
            if cv_mse<min_mse:
                min_mse=cv_mse
                best=[x1,x2,min_mse]
#     print("交叉验证集MSE最佳均值为 %s" % (best[3]))
    print("params are:",best)


# In[ ]:


"""
Bayesian ridge and ridge
"""
print(len(train_x_try_2.columns))
kfold_scores_v6(train_x_try_scaled_2,train_y_try)#bayesian ridge and ridge


# In[ ]:


"""
用八个，最佳
"""


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,LinearRegression,Ridge
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingRegressor
from lightgbm import LGBMRegressor

para_xgb={'learning_rate': 0.07333333333333333, 'n_estimators': 500, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.4, 'gamma': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
para_lgb={'num_leaves': 10, 'max_depth': 3, 'learning_rate': 0.025833333333333333, 'n_estimators': 400, 'min_child_weight': 1, 'subsample': 0.1, 'colsample_bytree': 0.4, 'nthread': 4, 'objective': 'regression'}
alpha_ridge=0.06

alg1 = xgb.XGBRegressor(**para_xgb)
alg2 = LGBMRegressor(**para_lgb)
alg3=Ridge(alpha_ridge)


####1SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####4GBRT回归####
from sklearn import ensemble
para_gbr={'learning_rate': 0.020000000000000004, 'loss': 'huber', 'max_depth': 5, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 300, 'random_state': 0, 'subsample': 0.09000000000000001}
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(**para_gbr)
####5BayesianRidge贝叶斯岭回归
from sklearn.linear_model import BayesianRidge,TheilSenRegressor
model_BayesianRidge = BayesianRidge()
####6TheilSen泰尔森估算
model_TheilSenRegressor = TheilSenRegressor(n_jobs=2)
# 7
from sklearn.kernel_ridge import KernelRidge
model_KernelRidge=KernelRidge(alpha=0.06, kernel='polynomial', degree=3, coef0=1)
regressors = [alg1,alg2,alg3,model_KernelRidge,model_SVR,model_TheilSenRegressor,model_BayesianRidge,model_GradientBoostingRegressor]


# In[ ]:


# 看看每个单独模型的预测结果
results_of_models=pd.DataFrame()
# regressors = [alg1,alg2,alg3,model_KernelRidge,model_SVR,model_TheilSenRegressor,model_BayesianRidge,]
i=0
for model in regressors:
    print(model)
    kfold_scores(model,train_x_try_scaled_2,train_y_try)
    model.fit(train_x_try_scaled_2,train_y_try)
    mean_squared_error(model.predict(train_x_try_scaled_2),train_y_try)
    result=model.predict(zhengqi_test_try_scaled)
    results_of_models[str(i)]=result
    i=i+1


# In[1]:


def kfold_scores_v8(x_train,y_train,regressors):
    kf = KFold(n_splits = 5, random_state= 1, shuffle=True)
    best = []
    kfold_results=[]
    record_of_index_train=[]
    record_of_index_test=[]
    min_mse=10
    turn=0
    for x0 in np.linspace(0.1,1,5):
        for x1 in np.linspace(0.1,1,5):
            for x2 in np.linspace(0.1,1,5):
                for x3 in np.linspace(0.1,1,5):
                    for x4 in np.linspace(0.1,1,5):
                        for x5 in np.linspace(0.1,1,5):
                            for x6 in np.linspace(0.1,1,5):
                                for x7 in np.linspace(0.1,1,5):
                                    predict_y = []
                                    if turn==0:               
                                        for kf_train,kf_test in kf.split(x_train):
                                            record_of_index_train.append(kf_train)
                                            record_of_index_test.append(kf_test)
                                            predictions=pd.DataFrame()
                                            j=0
                                            for model in regressors:
                                                model.fit(x_train.iloc[kf_train],y_train.iloc[kf_train])
                                                one_model_prediction=model.predict(x_train.iloc[kf_test])
                                                predictions[str(j)]=one_model_prediction
                                                j=j+1
                                            kfold_results.append(predictions)
                                        for i,prediction in enumerate(kfold_results):
                                            y_pred_train=(prediction.iloc[:,0]*x0+prediction.iloc[:,1]*x1+prediction.iloc[:,2]*x2+prediction.iloc[:,3]*x3+prediction.iloc[:,4]*x4+prediction.iloc[:,5]*x5
                                            +prediction.iloc[:,6]*x6+prediction.iloc[:,7]*x7)/(x0+x1+x2+x3+x4+x5+x6+x7)
                                            kf_test=record_of_index_test[i]
                                            mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
                                            predict_y.append(mse)
                                    else:#以后只直接调用第一次算好的结果)
                                        for i,prediction in enumerate(kfold_results):
                                            y_pred_train=(prediction.iloc[:,0]*x0+prediction.iloc[:,1]*x1+prediction.iloc[:,2]*x2+prediction.iloc[:,3]*x3+prediction.iloc[:,4]*x4+prediction.iloc[:,5]*x5
                                            +prediction.iloc[:,6]*x6+prediction.iloc[:,7]*x7)/(x0+x1+x2+x3+x4+x5+x6+x7)
                                            kf_test=record_of_index_test[i]
                                            mse = mean_squared_error(y_train.iloc[kf_test],y_pred_train)
                                            predict_y.append(mse)
                                    assert len(predict_y)==5
                                    cv_mse=np.mean(predict_y)
                                    print("current turn mse:",cv_mse,"turn is",turn)
                                    print(best)
                                    turn=turn+1
                                    if cv_mse<min_mse:
                                        min_mse=cv_mse
                                        best=[x0,x1,x2,x3,x4,x5,x6,x7,min_mse]
#     print("交叉验证集MSE最佳均值为 %s" % (best[3]))
    print("params are:",best)


# In[ ]:


kfold_scores_v8(train_x_try_scaled_2,train_y_try_2,regressors)


# In[ ]:


print(zhengqi_test_try_2.columns)
print(train_x_try_2.columns)
alg1.fit(train_x_try_scaled_2,train_y_try_2)
alg2.fit(train_x_try_scaled_2,train_y_try_2)
alg3.fit(train_x_try_scaled_2,train_y_try_2)
model_KernelRidge.fit(train_x_try_scaled_2,train_y_try_2)
model_SVR.fit(train_x_try_scaled_2,train_y_try_2)
model_TheilSenRegressor.fit(train_x_try_scaled_2,train_y_try_2)
model_BayesianRidge.fit(train_x_try_scaled_2,train_y_try_2)
model_GradientBoostingRegressor.fit(train_x_try_scaled_2,train_y_try_2)
r1=alg1.predict(zhengqi_test_try_scaled_2)
r2=alg2.predict(zhengqi_test_try_scaled_2)
r3=alg3.predict(zhengqi_test_try_scaled_2)
r4=model_KernelRidge.predict(zhengqi_test_try_scaled_2)
r5=model_SVR.predict(zhengqi_test_try_scaled_2)
r6=model_TheilSenRegressor.predict(zhengqi_test_try_scaled_2)
r7=model_BayesianRidge.predict(zhengqi_test_try_scaled_2)
r8=model_GradientBoostingRegressor.predict(zhengqi_test_try_scaled_2)


# In[2]:


# 根据上面的权重，计算最终结果
ultimate_result=(0.1*r1+0.55*r2+1*r3+ 0.1*r4+ 0.1*r5 +0.55*r6 +1*r7 +0.55*r8)/(0.1+0.55+ 1.0+ 0.1+ 0.1+ 0.55+ 1.0+ 0.55)
np.savetxt("eight_model_stacked_without511172227_with_feature_combined_no_binned_scaled.txt",ultimate_result)


# In[ ]:


"""
看看不同方案的预测结果之间分布
"""
no_transformed=np.loadtxt("eight_model_stacked_without511172227_no_feature_combined_no_binned_scaled.txt")
transformed=np.loadtxt("eight_model_stacked_without511172227_with_feature_combined_with_binned_scaled.txt")
plt.figure(figsize=(400,26))
plt.xlabel("index")
plt.ylabel("values")
# plt.plot(range(1,len(y_pred_train_svr_ridge)+1), y_pred_train_svr_ridge)
plt.scatter(range(1,len(y_pred_train_svr_ridge)+1), y_pred_train_svr_ridge,marker='x',color='red')
# plt.plot(range(1,len(y_pred_train_svr_ridge)+1), y_pred_train_svr_ridge_xgb)
plt.scatter(range(1,len(y_pred_train_svr_ridge)+1), y_pred_train_svr_ridge_xgb,marker='o',color='blue')
plt.scatter(range(1,len(y_pred_train_svr_ridge)+1), y_pred_train2,marker='v',color='yellow')
plt.scatter(range(1,len(y_pred_train_svr_ridge)+1), no_transformed,marker='o',color='green')
plt.scatter(range(1,len(y_pred_train_svr_ridge)+1), transformed,marker='v',color='brown')
plt.legend(['svr_ridge','svr_ridge_xgb','ridge','e1','e2'])


# plt.xticks(np.arange(1, rounds+1, 1.0))
plt.show()

