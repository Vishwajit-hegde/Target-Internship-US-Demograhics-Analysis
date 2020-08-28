import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from cat_wise_clustering import create_category_data

def get_attributes(df,attr_type = 'PE'):
    if attr_type=='PE':
        percent_cols = []
        for col in df.columns:
            if(col[-2:]=='PE'):
                percent_cols.append(col)
        return df[percent_cols]
    else:
        count_cols = []
        for col in df.columns:
            if (col[-1]=='E') and (col[-2:]!='PE'):
                count_cols.append(col)
        return df[count_cols]

def drop_PR_attributes(df):
    PR_zipcodes = df[df.iloc[:,0].isna()].index
    PR_attributes = [x for x in df.columns if x[4:6]=='PR']
    nonPR_attributes = [x for x in df.columns if x[4:6]!='PR']
    for zipcode in PR_zipcodes:
        for attr1,attr2 in zip(PR_attributes,nonPR_attributes):
            df.loc[zipcode,attr2] = df.loc[zipcode,attr1]
    for attr in PR_attributes:
        df.drop(attr,axis=1,inplace=True)
    return df

def manual_attr_selection(data):
    classes = ['Households','Relationship','Marital Status','School Enrollment','Education Attainment','Ancestry',
               'Employment Status','Occupation','Industry','Class of Worker','Income','Sex and Age','Race']
    class_df = pd.Series(index=data.columns)
    col_list = list(data.columns)
    for col in col_list:
        if(col[:4]=='DP02' and int(col[6:9])>0 and int(col[6:9])<=16):
            class_df[col] = 'Households'
        elif(col[:4]=='DP02' and int(col[6:9])>16 and int(col[6:9])<=23):
            class_df[col] = 'Relationship'
        elif(col[:4]=='DP02' and int(col[6:9])>23 and int(col[6:9])<=35):
            class_df[col] = 'Marital Status'
        elif(col[:4]=='DP02' and int(col[6:9])>51 and int(col[6:9])<=57):
            class_df[col] = 'School Enrollment'
        elif(col[:4]=='DP02' and int(col[6:9])>57 and int(col[6:9])<=67):
            class_df[col] = 'Education Attainment'
        elif(col[:4]=='DP02' and int(col[6:9])>109 and int(col[6:9])<=121):
            class_df[col] = 'Language Spoken'
        elif(col[:4]=='DP02' and int(col[6:9])>122 and int(col[6:9])<=149):
            class_df[col] = 'Ancestry'
        elif(col[:4]=='DP03' and int(col[6:9])>0 and int(col[6:9])<=17):
            class_df[col] = 'Employment Status'
        elif(col[:4]=='DP03' and int(col[6:9])>25 and int(col[6:9])<=31):
            class_df[col] = 'Occupation'
        elif(col[:4]=='DP03' and int(col[6:9])>31 and int(col[6:9])<=45):
            class_df[col] = 'Industry'
        elif(col[:4]=='DP03' and int(col[6:9])>45 and int(col[6:9])<=50):
            class_df[col] = 'Class of Worker'
        elif(col[:4]=='DP03' and int(col[6:9])>51 and int(col[6:9])<=62):
            class_df[col] = 'Income'
        elif(col[:4]=='DP05' and int(col[6:9])>0 and int(col[6:9])<=32):
            class_df[col] = 'Sex and Age'
        elif(col[:4]=='DP05' and int(col[6:9])>35 and int(col[6:9])<=71):
            class_df[col] = 'Race'
    class_df = class_df.reset_index()
    class_df.rename({0:'class','index':'ID'},axis=1,inplace=True)
    data_dict = {}
    for cl in classes:
        data_dict[cl] = data[class_df[class_df['class']==cl]['ID'].values]
        data_dict[cl] = data_dict[cl].drop(data_dict[cl].columns[data_dict[cl].mean()>100],axis=1)
        data_dict[cl] = drop_zero_percent_attributes(data_dict[cl],pop_threshold=5)
        data_dict[cl] = drop_correlated_attributes(data_dict[cl],0.8)
    col_list = []
    for cl in classes:
        col_list += list(data_dict[cl].columns)
    return data[col_list]

def drop_zero_pop_zipcodes(df):
    zero_indexes = df[df.iloc[:,0]==0].index
    return df.drop(list(zero_indexes))

def fill_missing_values(df,data,attr_type='PE'):
    for col in df.columns[(df<0).all()]:
        col_ = col[:9]+attr_type
        df[col] = data[col_]
    for col in df.columns:
        df.loc[df[col]<0,col] = np.nan
    return df.fillna(df.median())

def drop_correlated_attributes(df,corr_threshold=0.8):
    corr_df = df.corr()
    correlation_dict = {}
    for col in df.columns:
        correlation_dict[col] = list(corr_df.loc[(abs(corr_df[col])>corr_threshold) & (corr_df[col]!=1),col].index)
    drop_cols = []
    for col in df.columns:
        if col not in drop_cols:
            drop_cols += correlation_dict[col]
            drop_cols = list(np.unique(drop_cols))
    return df.drop(drop_cols,axis=1)

def drop_zero_percent_attributes(df,z_count=20000,pop_threshold=500,pop_threshold_count=1000):
    drop_cols = []
    for col in df.columns:
        zero_count = len(df[df[col]==0][col])
        five_count = len(df[df[col]>=pop_threshold][col])
        if(zero_count>z_count and five_count<pop_threshold_count):
            drop_cols.append(col)
    return df.drop(drop_cols,axis=1)
