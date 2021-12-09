# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 07:40:54 2021

@author: valteresj
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
import statsmodels.api as sm


def distance_matrix(A, B, squared=False):

    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def calc_distance(x1,y1,a,b,c):
    d=np.abs((a*x1+b*y1+c))/(np.sqrt(a*a+b*b))
    return d

def best_k(dt):
    wcss=[]
    min_k=1
    max_k=10
    vet=np.arange(min_k,max_k,1)
    for i in range(min_k,max_k):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dt)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
        
    a=wcss[0]-wcss[len(wcss)-1]
    b=max_k-min_k
    c1=min_k*wcss[len(wcss)-1]
    c2=max_k*wcss[0]
    c=c1-c2
    distance_of_point=[]
    i=1
    for i in range(max_k-1):
        distance_of_point.append(calc_distance(vet[i],wcss[i],a,b,c))
    
    return np.argmax(distance_of_point)+1

def generate_feature_kmeans(dt,fit_model=None):
    dt_preprocessing=dt.copy()
    scaler = MinMaxScaler()
    scaler.fit(dt_preprocessing.values)
    dt_minmax=scaler.transform(dt_preprocessing.values)
    if fit_model is None:
        size_k=best_k(dt_minmax)
        kmeans = KMeans(size_k)
        kmeans.fit(dt_minmax)
    else:
        kmeans=fit_model
    centroids=kmeans.cluster_centers_
    
    dist=distance_matrix(dt_minmax,centroids)
    for i in range(dist.shape[1]):
        dt['kmeans_'+str(i)]=dist[:,i]
    return kmeans


def expansion_feat(dt,alvo=None,fit_model=None,n_pol=2):
    
    if fit_model is None:
        alvo_mod=alvo.astype('category').cat.codes
        X=dt.values
        model= sm.GLM(alvo_mod,X,family=sm.families.Binomial())
        model_results = model.fit()
        parms_=model_results.params
    else:
        parms_=fit_model
    Test=parms_.values*dt.values
    poly = PolynomialFeatures(n_pol)
    Test=poly.fit_transform(Test)
    Test=np.delete(Test, list(range(dt.values.shape[1]+1)), axis=1)
    for idx,i in enumerate(range(Test.shape[1])):
        dt['Expansion_'+str(idx)]=Test[:,idx]
    return parms_



class AEF(object):
    def __init__(self,n_pol=2,target=None,dt=None,path=None):
        self.n_pol=n_pol
        self.target=target
        self.dt=dt
        self.path=path
        
    
    def auto_eng_features(self):
        if self.target is not None:
            alvo=self.dt[self.target]
            dt_study=self.dt.drop(columns=self.target).copy()
            
        self.dt=self.dt.reset_index(drop=True)
        dt_study=dt_study.reset_index(drop=True)

        dt_study_mod=pd.get_dummies(dt_study)
        
        var_ref=dt_study_mod.columns
        
      
        fit_=generate_feature_kmeans(dt_study_mod)
        
        feat_expan=expansion_feat(dt=dt_study_mod,alvo=alvo,n_pol=self.n_pol)
        
        var_diff=list(dt_study_mod.columns[np.isin(np.array(dt_study_mod.columns),np.array(var_ref),invert=True)])
        dt_study=pd.concat([self.dt,dt_study_mod.loc[:,var_diff]],axis=1)
        
        output= {'table_final':dt_study,'fit_expansion':feat_expan,'fit_kmeans':fit_}
        return output
    
    def pred(self,dt_tst,params=None):
        if self.target is not None:
            dt_study=dt_tst.drop(columns=self.target).copy()
            
        dt_tst=dt_tst.reset_index(drop=True)
        dt_study=dt_study.reset_index(drop=True)

        dt_study_mod=pd.get_dummies(dt_study)
        
        var_ref=dt_study_mod.columns
        
        generate_feature_kmeans(dt_study_mod,params['fit_kmeans'])
        
        expansion_feat(dt=dt_study_mod,fit_model=params['fit_expansion'])
        
        var_diff=list(dt_study_mod.columns[np.isin(np.array(dt_study_mod.columns),np.array(var_ref),invert=True)])
        dt_study=pd.concat([dt_tst,dt_study_mod.loc[:,var_diff]],axis=1)
        
        return dt_study
        
        
        