# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:15:10 2023

@author: hp
"""

#algomative techinue is applicable for small dataseat
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#Now import file from data
Univ1=pd.read_excel("C:/2-Datasets/University_Clustering.xlsx")
Univ1=pd.read_excel("University_Clustering.xlsx")
a=Univ1.describe()

#we have column "State" which really not useful we will drop it
Univ=Univ1.drop(["State"],axis=1)
#we know that there is scale difference among the columns,which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to Univ dataframe 
#for all the rows and column from 1 until end
#since 0th column has University name hence skipped
df_norm=norm_func(Univ.iloc[:,1:])
#you can check df_norm dataframe which is scaled
#between values from 0 to 1
#you can apply describe function to new dataframe
b=df_norm.describe()

#before you apply clustering,u need to plot dendrogram first
#now to create dendrogram,we need to measure distance

#we have to import linkage

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#linkage function give us hierarchical or aglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")

plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")

#ref help of dendrogram
#sch.dendrogram(z)

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#dendrogram

#applying agglomerative clustering choosing 3 as clusters
#from dendrogram
#whatever has been displayed in dendrogram is not clustering
#it is just showing numbe of possible cluster
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)
#apply labels to clusters
h_complete.labels_cluster_labels=pd.series(h_complete.labels_)
cluster_labels=pd.Series(h_complete.labels_)
#Assign this series to Univ dataframe as column and name the column
Univ['clust']=cluster_labels#we want to relocate column 7 to 0 th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check Univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
#from o/p cluster has 2 got highest top10
#lowest accepts ratio,best faculty ratio and highest expenses
#highest graduates ratio
Univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()