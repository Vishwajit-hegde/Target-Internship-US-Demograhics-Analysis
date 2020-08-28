import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
sys.path.append('/Users/vishwajit/.pyenv/versions/3.7.3/lib/python3.7/site-packages/')
import umap
from tqdm import tqdm
sys.path.append('/Users/vishwajit/Desktop/SOMPY_robust_clustering-master/')
import sompy
from sompy.sompy import SOMFactory
import plotly.express as px

def scale_data(data,scale):
     if scale=='minmax':
         X = MinMaxScaler().fit_transform(data)
     elif scale=='standard':
         X = MinMaxScaler().fit_transform(data)
     else:
         X = data.values
     return X

def KMeans_clustering(data,min_clust,max_clust,freq,scale='minmax',plot_inertia=True):
    X = scale_data(data,scale)
    inertia = []
    for clust in tqdm(np.arange(min_clust,max_clust,freq)):
        km = KMeans(clust).fit(X)
        inertia.append(km.inertia_)
    if plot_inertia:
        plt.figure(figsize=(15,10))
        plt.plot(np.arange(min_clust,max_clust,freq),inertia)
        plt.xlabel('Number of clusters',fontsize=18)
        plt.ylabel('K-Means Inertia',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig('KMeans Inertia Plot',dpi=200)
        plt.show()

def DBSCAN_clustering(data,eps_list,min_s_list,scale='minmax'):
    X = scale_data(data,scale)
    min_s = 5
    opt_eps = eps_list[0]
    max_s = -1
    for eps in eps_list:
        print(f'epsilon = {eps}')
        db = DBSCAN(eps,min_s).fit(X)
        n_clust = len(set(db.labels_))-1
        print(f'number of clusters = {n_clust}')
        noise_pts = np.sum([1 if x==-1 else 0 for x in db.labels_])
        print(f'number of noisy points = {noise_pts}')
        if n_clust>0:
            s_score = silhouette_score(X,db.labels_)
            print(f'silhouette score = {s_score}')
            max_s = max(s_score,max_s)
            if max_s == s_score:
                opt_eps = eps
    opt_min_s = 5
    for min_s in min_s_list:
        print(f'min samples = {min_s}')
        db = DBSCAN(opt_eps,min_s).fit(X)
        n_clust = len(set(db.labels_))-1
        print(f'number of clusters = {n_clust}')
        noise_pts = np.sum([1 if x==-1 else 0 for x in db.labels_])
        print(f'number of noisy points = {noise_pts}')
        if n_clust>0:
            s_score = silhouette_score(X,db.labels_)
            print(f'silhouette score = {s_score}')
            max_s = max(s_score,max_s)
            if max_s == s_score:
                opt_min_s = min_s
    db = DBSCAN(opt_eps,opt_min_s).fit(X)

    s_score = silhouette_score(X,db.labels_)
    print(f'optimum number of clusters using dbscan = {n_clust}')
    print(f'optimum silhouette score using dbscan = {s_score}')
    clust_series = pd.Series(db.labels_).value_counts()
    cluster_std = np.std(clust_series.values)
    print(f'the standard deviation in the number of clusters = {cluster_std}')

def SOM_clustering(data,grid_size_list,scale='minmax',plot_grid=True,n_clusters=None):
    X = scale_data(data,scale)
    terror = {}
    sm = {}
    for mapsize in grid_size_list:
        print(f'grid size = {mapsize}')
        sm[str(mapsize)] = SOMFactory().build(X, normalization = 'var', initialization='pca',mapsize=mapsize)
        sm[str(mapsize)].train(n_job=1, verbose=False, train_rough_len=4, train_finetune_len=10)
        quant_error = np.array(sm[str(mapsize)]._quant_error)[-1:,2][0]
        topo_error = sm[str(mapsize)].calculate_topographic_error()
        terror[str(mapsize)] = topo_error
        print(f'quantization error = {quant_error}')
        print(f'topographical error = {topo_error}')


    min_terror = 1
    for mapsize in grid_size_list:
        min_terror = min(min_terror,terror[str(mapsize)])
        if(min_terror == terror[str(mapsize)]):
            opt_mapsize = mapsize

    sm = sm[str(opt_mapsize)]
    x = sm.project_data(X)
    if n_clusters:
        print(f'number of clusters = {n_clusters}')
        labels,_,_ = sm.cluster(opt = n_clusters, cl_type = 'kmeans')
        cluster_labels = []
        for i in range(X.shape[0]):
            cluster_labels.append(labels[x[i]])
        s_score = silhouette_score(X,cluster_labels)
        print(f'silhouette score = {s_score}')
        if plot_grid:
            image_data = labels.reshape(60,60)
            plt.figure(figsize=(25,15))
            plt.imshow(image_data,cmap='viridis')
            plt.grid()
            plt.savefig('SOM_cluster_map.png',dpi=200)
    else:
        max_s = -1
        labels = {}
        for clust in [100,150,200,250,300]:
            print(f'number of clusters = {clust}')
            labels[clust],_,_ = sm.cluster(opt = clust, cl_type = 'kmeans')
            cluster_labels[clust] = []
            for i in tqdm(range(X.shape[0])):
                cluster_labels[clust].append(labels[clust][x[i]])
            s_score = silhouette_score(X,cluster_labels[clust])
            print(f'silhouette score = {s_score}')
            max_s = max(max_s,s_score)
            if(max_s==s_score):
                opt_clust = clust
        print(f'optimum number of clusters = {opt_clust}')
        if plot_grid:
            image_data = labels.reshape(60,60)
            plt.figure(figsize=(25,15))
            plt.imshow(image_data,cmap='viridis')
            plt.grid()
            plt.savefig('SOM_cluster_map.png',dpi=200)

def TSNE_dim_reduction(data, perplexity_list, pca_evr=0.95, scale='minmax'):
    X = scale_data(data,scale)
    n = X.shape[0]
    pca_transformed = PCA(0.95).fit_transform(X)
    tsne = {}
    S = {}
    for p in tqdm(perplexity_list):
        tsne[p] = TSNE(perplexity=p).fit(pca_transformed)
        S[p] = 2*tsne[p].kl_divergence_ + np.log(n)/n * p
    opt_p = min(S,key=S.get)
    print(f'optimum perplexity = {opt_p}')
    return tsne, opt_p

def UMAP_dim_reduction(data, n=5, m=0.01, s=0.5, scale='minmax', plot=True):
    X = scale_data(data,scale)
    embedding = umap.UMAP(n_components=2,
                          n_neighbors=n,
                          min_dist=m,
                          metric='euclidean',
                          random_state=1,
                          spread=s,
                          n_epochs=1000).fit_transform(X)
    return embedding
