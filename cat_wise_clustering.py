import sys
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import data_preprocessing as dp
sys.path.append('/Users/vishwajit/.pyenv/versions/3.7.3/lib/python3.7/site-packages/')
import umap
from tqdm import tqdm
from yellowbrick.cluster import KElbowVisualizer
from xgboost import XGBClassifier
sys.path.append('/Users/vishwajit/Desktop/SOMPY_robust_clustering-master/')
import sompy
from sompy.sompy import SOMFactory

def create_category_data(data,classes,save_data=False,corr_threshold=0.8,drop_pop_counts=True):
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
        if drop_pop_counts:
            data_dict[cl] = data_dict[cl].drop(data_dict[cl].columns[data_dict[cl].mean()>100],axis=1)
        data_dict[cl] = dp.drop_correlated_attributes(data_dict[cl],corr_threshold)
        data_dict[cl] = dp.drop_zero_percent_attributes(data_dict[cl],pop_threshold=5)
        if save_data:
            data_dict[cl].to_csv(cl+'_data.csv')
    return data_dict

def cluster_category_data(df,scale_data='minmax',dim_red_method='som',use_elbow_method='True',cluster_method='hierarchical',n_clusters=None,verbose=1,perplexity=None):
    """
    :param df: dataframe containing all the columns belonging to a category to be used in clustering
    :param scale_data: method to be used to scale the dataset
    :param dim_red_method: options are 'som', 'umap', 'tsne', None. If  None, do clustering directly.
    :param use_elbow_method: if True, elbow method is used to find the optimum number of clusters. If False, n_clusters needs to be specified
    :param cluster_method: options are 'kmeans' and 'hierarchical'. In either case kmeans is used for the elbow method(because of the time required).
    :param n_clusters: If use_elbow_method is False, n_clusters needs to be given.
    :param verbose: If True, output the progress in clustering process
    :param perplexity: If method used is TSNE, perplexity nedds to be specified
    """
    t = time.time()

    if scale_data=='minmax':
        X = MinMaxScaler().fit_transform(df)
    elif scale_data=='standard':
        X = StandardScaler().fit_transform(df)
    else:
        X = df.values

    if verbose:
        print(f'number of features = {df.shape[1]}')

    if dim_red_method=='som':
        if verbose:
            print('Self Organising Maps is being used for dimensionality reduction...')
        opt_k = 2
        max_s = -1
        f = 0
        for mapsize in [(30,30)]:
            if verbose:
                print(f'map size = {mapsize}')
            sm = SOMFactory().build(X, normalization='var', initialization='pca',mapsize=mapsize)
            sm.train(n_job=1, verbose=False, train_rough_len=100, train_finetune_len=500)
            if use_elbow_method:
                model = KElbowVisualizer(KMeans(),k=20,timings=False)
                elbow = model.fit(sm.codebook.matrix).elbow_value_
                if elbow and verbose:
                    print(f'elbow value = {elbow}')
                if not elbow:
                    if verbose:
                        print('elbow not found')
                    ms = -1
                    for k in range(2,20):
                        km_labels = KMeans(k).fit_predict(sm.codebook.matrix)
                        s = silhouette_score(sm.codebook.matrix,km_labels)
                        if s>ms:
                            elbow = k
            else:
                elbow = n_clusters
            x = sm.project_data(X)
            labels,_,_ = sm.cluster(opt=elbow,cl_type=cluster_method)
            clabels = []
            for i in range(X.shape[0]):
                clabels.append(labels[x[i]])
            s_score = silhouette_score(X,clabels)
            if verbose:
                print(f'silhouette score = {round(s_score, 3)}')
            max_s = max(s_score,max_s)
            if(max_s==s_score):
                opt_k = elbow
                opt_labels = clabels
                opt_size = mapsize
            if(max_s>s_score):
                break
        if verbose:
            print(f'optimum mapsize = {opt_size}')
            print(f'optimum number of clusters = {opt_k} & silhouette score = {round(max_s,3)}')
            print(f'time taken = {round(time.time()-t,1)}')
        return opt_labels, opt_k

    elif dim_red_method:
        if dim_red_method=='umap':
            print('UMAP is being used for dimensionality reduction...')
            embedding = umap.UMAP(n_components=2,
                                  n_neighbors=5,
                                  min_dist=0.0001,
                                  metric='euclidean',
                                  random_state=1,
                                  spread=0.5,
                                  n_epochs=1000).fit_transform(X)
            print('UMAP embedding done...')
        elif dim_red_method=='tsne':
            print('t-SNE is being used for dimensionality reduction...')
            embedding = TSNE(perplexity=perplexity).fit_transform(X)
            print('t-SNE embedding is done...')
        if use_elbow_method:
            model = KElbowVisualizer(KMeans(),k=20,timings=False)
            elbow = model.fit(embedding).elbow_value_
        else:
            elbow = n_clusters
        if cluster_method=='kmeans':
            opt_labels = KMeans(elbow).fit_predict(embedding)
        elif cluster_method=='hierarchical':
            opt_labels = AgglomerativeClustering(elbow).fit_predict(embedding)
        if verbose:
            s_score = silhouette_score(X,opt_labels)
            print(f'number of clusters = {elbow} and silhouette_score = {s_score}')
        return opt_labels,elbow

    else:
        if use_elbow_method:
            model = KElbowVisualizer(KMeans(),k=20,timings=False)
            elbow = model.fit(X).elbow_value_
        else:
            elbow = n_clusters
        if cluster_method=='kmeans':
            opt_labels = KMeans(elbow).fit_predict(X)
        elif cluster_method=='hierarchical':
            opt_labels = AgglomerativeClustering(elbow).fit_predict(X)
        print(f'silhouette score = {round(silhouette_score(X,opt_labels),3)}')
        return opt_labels, elbow

def get_feature_importance(data,labels,display=True):
    """
    :param data: dataframe to be used for feature importance
    :param labels: cluster labels to be used for classification
    :param display: Number of top important features and respective feature importance to be displayed.
    """
    df = pd.DataFrame(MinMaxScaler().fit_transform(data),index=data.index,columns=data.columns)
    imp_dict = {}
    for c in set(labels):
        print(f'cluster id = {c}')
        y = [1 if x==c else 0 for x in labels]
        X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=10)
        clf = XGBClassifier(n_estimators=1000,
                            max_depth=6,
                            learning_rate=0.01,
                            objective='binary:logistic',
                            eval_metric='auc')
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print('accuracy score is ',accuracy_score(y_test,y_pred))
        imp_dict[c] = clf.get_booster().get_score(importance_type='gain')
        if display:
            feature_imp_series = pd.Series(imp_dict[c],index=data.columns)
            print(feature_importance_df[cl+'_'+str(c)].dropna().sort_values(ascending=False)[:display])
    return imp_dict

def validate_clusters(df,labels):
    y = cluster_labels
    X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=10)
    clf = XGBClassifier(n_estimators=1000,
                        max_depth=6,
                        learning_rate=0.01,
                        objective='binary:logistic',
                        eval_metric='auc')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(f'accuracy score = {round(accuracy_score(y_test,y_pred),3)}')

def find_optimum_clusters(data,start,stop,step,clust_method='kmeans'):
    max_s = -1
    opt_k = start
    for c in np.arange(start,stop,step):
        if clust_method=='kmeans':
            model = KMeans(c).fit(data)
        elif clust_method=='hierarchical':
            model = AgglomerativeClustering(c).fit(data)
        s_score = silhouette_score(data,model.labels_)
        max_s = max(s_score,max_s)
        if max_s==s_score:
            opt_k = c
    return opt_k
