#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style


# In[2]:


data = pd.read_csv("iris.csv",index_col=0)
data.shape


# In[3]:


data.head()


# In[4]:


from sklearn.cluster import KMeans
X=data.iloc[:,:4].values
avg_distance=[]
for i in range(1,11):
   clusterer=KMeans(n_clusters=i,random_state=2).fit(X)
   avg_distance.append(clusterer.inertia_)   


# In[5]:


plt.plot(range(1,11), avg_distance)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distance")
plt.show()


# In[6]:


kmeans=KMeans(n_clusters=2,random_state=2)
y_means=kmeans.fit_predict(X)


# In[7]:


plt.figure(figsize=[10,8])
plt.scatter(X[y_means == 0,0], X[y_means == 0,1], 
            s = 100, c = "red", label = 'cluster 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], 
            s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'black', label = 'Centroids')

plt.legend()
plt.show()

