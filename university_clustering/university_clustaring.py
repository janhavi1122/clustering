# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 08:57:46 2023

@author: santo
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#now import file from dataset and create dataset

univ1=pd.read_excel("E:/datascience/clustaring/University_Clustering.xlsx")
univ1

a=univ1.describe()

#we have one column "state which really not useful we will drop it
univ=univ1.drop(["State"],axis=1)
#there is scale difference among the column
#either by using the normallization or standardization
#whenever there is mixed data apply to normalization

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now we apply this normalization function to univ dataframe for all the row
#all the rows and column until 1 end
#since 0th column has university name as skipped

df_norm=norm_func(univ.iloc[:,1:])
#you can check the df_norm dataframe which is scaled
#between value from 0,1
b=df_norm.describe()

#before you apply clustering ,plot the dendrogram 
#now create dendrogram ,we measure the distance
#import the linkage
from scipy.cluster.hierachy import linkage
import scipy.cluster.hierarchy as sch
#linkage function give us hierchical or aglomerative
#ref the help for linkage

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram");
plt.xlabel("Index")
plt.ylabel("Distance")
#ref help of dendrogram
#sch.dendrogram(z)
sch.denrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendrogram()
#applying agglomerative clustering choosing 3 as cluster
from sklearn.cluster import AgglomerativeClustering

# Create an AgglomerativeClustering model
h_complete = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean')

# Fit the model to the normalized data
cluster_labels = h_complete.fit_predict(df_norm)

# Assign the cluster labels to your 'univ' DataFrame
univ['clust'] = cluster_labels

#we want to relocate the column 7 to 0 th position
Univ1=univ.iloc[:,[7,1,2,3,4,5,6]]
#check the univ1 dataframe
Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
#from the output cluster 2 has got highest

Univ1.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()
#___________________________________________________________________________________


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
#let us try to understand first how k means work for two dimensional data
#for that , generate random no in the range 0,1 
# and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 colmuns
df_xy=pd.DataFrame (columns=['X','Y'])
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x='X',y='Y',kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)
'''
with data X and Y apply KMeans model generate scatter plot with scale /font = 10
cmap=plt.cm.cool color combination

'''
model1.labels
df_xy.plot(x='X',y='Y',c=model1.labels_,kind='scatter',s=10,camp=plt.cm.coolwarm)

Univ1=pd.read_excel("E:/datascience/clustaring/University_Clustering.xlsx")
Univ1.describe()
Univ1.drop(["State"],axis=1)
'''
 we know that there is scale difference among the column which we have either by 
using normalization or stadarization 
'''
def norm_func(i):
    x=(i-i.min())/(i-max()-i.min())
# Now apply this normalization function to univ dataframe for ALL the row 
df_norm=norm_func(Univ1.iloc[:,1:])
'''
    what will ideal cluster no will be 1,2,3
    '''
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)#total with in sum of square
    '''
    '''
TWSS
#As k increases the TWSS value decreases
plt.plot(k,TWSS,'ro-');
plt.xlabel('No_of_clusters');
plt.ylabel('Toatal_within_ss')

'''
How to select value of k from elbow curve when k changes for 2 to 3, then decreases in twss 
is heigher than k when changes from 3 to 4 .
when k value changes from 5 to 6 decreases in twss is considardly less, hence considered k=3
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
Univ1['clust']=mb
Univ1.head()
Univ=Univ1.iloc[:,[7,0,1,2,3,4,5,6]]
Univ
Univ.iloc[:,2:8].groupby(Univ.clust).mean()
Univ.to_csv('kmeans_university.csv',encoding='utf-8')
import os
    

#___________________________________________________________________________________
#algomative techinue is applicable for small dataseat
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#Now import file from data
Univ1=pd.read_excel("E:/datascience/clustaring/University_Clustering.xlsx")
Univ1=pd.read_excel("E:/datascience/clustaring/University_Clustering.xlsx")
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
Univ1.to_csv("E:/datascience/clustaring/University_Clustering.xlsx",encoding="utf-8")
import os
os.getcwd()






















