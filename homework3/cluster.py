import pandas as pd
import json
import matplotlib.pyplot as plt
# import os
# def data_manager():
#     data=open('./Tweets.txt','rb')
#     text=[]
#     cluster=[]
#     for line in data.readlines():
#         tmp=json.loads(line)
#         text.append(tmp['text'])
#         cluster.append(tmp['cluster'])
#     return text,cluster
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, MeanShift, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture


def gen_dataset():
    _data=open('./Tweets.txt','rb')
    dataset=pd.DataFrame(columns=['text','cluster'])
    _labels=set()
    for line in _data.readlines():
        tmp=json.loads(line)
        dataset.loc[len(dataset)]=[tmp['text'],tmp['cluster']]
        _labels.add(tmp['cluster'])
    tfidfv=TfidfVectorizer()
    _tiv_data=tfidfv.fit_transform(dataset['text'])
    return _tiv_data,dataset['cluster'],len(_labels)


def k_means(data,labels,n_clusters):
    '''
    硬聚类算法
    结果会变化！
    :param data:
    :param labels:
    :param n_clusters:
    :return:
    '''
    kms=KMeans(n_clusters=n_clusters)
    kms_res=kms.fit(data)
    kms_eval=normalized_mutual_info_score(labels,kms_res.labels_)
    print(kms_eval)
    return kms_eval

def spectral_clustering(data,labels,n_clusters):
    '''
    谱聚类
    结果不会变化！
    :param data:
    :param labels:
    :param n_clusters:
    :return:
    '''
    spc=SpectralClustering(n_clusters=n_clusters)
    spc_res=spc.fit(data)
    spc_eval=normalized_mutual_info_score(labels,spc_res.labels_)
    #print(spc_eval)
    return spc_eval

def avg_agglomerative_clustering(data,labels,n_clusters):
    '''
    从每一个点开始作为一个类，然后迭代的融合最近的类。能创建一个树形层次结构的聚类模型
    结果不会变化！
    :param data:
    :param labels:
    :param n_clusters:
    :return:
    '''
    agg=AgglomerativeClustering(n_clusters=n_clusters,linkage="average")
    agg_res=agg.fit(data.toarray())
    agg_eval=normalized_mutual_info_score(labels,agg_res.labels_)
    #print(agg_eval)
    return agg_eval

def ward_hierarchical_clustering(data,labels,n_clusters):
    agg=AgglomerativeClustering(n_clusters=n_clusters,linkage="ward")
    agg_res=agg.fit(data.toarray())
    agg_eval=normalized_mutual_info_score(labels,agg_res.labels_)
    return agg_eval

def affinity_propagation(data,labels):
    '''
    不需要设定聚类个数，聚类结果不会变化
    :param data:
    :param labels:
    :return:
    '''
    aff=AffinityPropagation()
    aff_res=aff.fit(data)
    aff_eval=normalized_mutual_info_score(labels,aff_res.labels_)
    #print(aff_eval)
    return aff_eval
def mean_shift(data,labels,bandwidth):
    '''
    均值漂移的不断迭代过程；
    :param data:
    :param cluster:
    :param bandwidth:
    :return:
    '''

    ms=MeanShift(bandwidth=bandwidth)
    ms_res=ms.fit(data.toarray())
    ms_eval=normalized_mutual_info_score(labels,ms_res.labels_)
    #print(ms_eval)
    return ms_eval
def density_based_clustering(data,labels,eps,minPts):
    '''
    Density-based Clustering
    :param eps:
    :param minPts:
    :return:
    '''
    dbc=DBSCAN(eps=eps,min_samples=minPts)
    dbc_res=dbc.fit(data.toarray())
    dbc_eval=normalized_mutual_info_score(labels,dbc_res.labels_)
    #print(dbc_eval)
    return dbc_eval

def gaussian_mixtures(data,clusters,n_components):
    mix=GaussianMixture(n_components=n_components,covariance_type='spherical')
    mix_res=mix.fit(data.toarray()).predict(data.toarray())
    mix_eval=normalized_mutual_info_score(clusters,mix_res)
    return mix_eval

def all_algorithms():
    res1=[[]for i in range(4)]
    data,labels,num=gen_dataset()
    bandwidth=estimate_bandwidth(data.toarray())
    print("bandwidth: ",bandwidth)
    for i in range(10):
        ##K-means
        kms_res=0
        for j in range(10):
            kms_res+=k_means(data,labels,85+i)
        res1[0].append(kms_res/10)
        ##spectral_clustering
        res1[1].append(spectral_clustering(data,labels,85+i))
        ##avg
        res1[2].append(avg_agglomerative_clustering(data,labels,85+i))
        ##ward
        res1[3].append(ward_hierarchical_clustering(data,labels,85+i))

    return res1


def plot_mean_shift():
    res1=[]
    res2=[]
    dataset, labels, num = gen_dataset()
    for i in range(10):
        res1.append(mean_shift(dataset,labels,(5+i)/10))
        res2.append(density_based_clustering(dataset,labels,(5+i)/10,1))
    return res1,res2


def plot(res):
    x1=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
    p1=plt.scatter(x1,res)
    plt.title("NMI-eps")
    plt.show()

def plot_gaussian():
    dataset,labels,num=gen_dataset()
    res=[]
    x=[]
    for i in range(20):
        x.append(65+i)
        tmp=0
        for j in range(5):
            tmp+=gaussian_mixtures(dataset,labels,65+i)
        res.append(tmp/5)
    p1=plt.scatter(x,res)
    plt.title("Gaussian-NMI-n_componts")
    plt.show()
def plotAcc():
    res=all_algorithms()
    x1=[85,86,87,88,89,90,91,92,93,94]
    p1 = plt.scatter(x1,res[0])
    p2 = plt.scatter(x1, res[1])
    p3 = plt.scatter(x1, res[2])
    p4 = plt.scatter(x1, res[3])
    #p5 = plt.scatter(x1, res[4])
    #p6 = plt.scatter(x1, res[5])
    #p7 = plt.scatter(x1, res[6])
    plt.legend(handles = [p1, p2,p3,p4,], labels = ['k-means', 'spectural-clustering','avg-agg','ward-agg'], loc = 'best')
    plt.title("NMI-nclusters")
    plt.show()

if __name__ == '__main__':
    #res1,res2=plot_mean_shift()
    #plot(res1)
    plotAcc()



