# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 07:33:12 2023

@author: 
"""


#Problem Statement:

#The dataset contains crime statistics for various states in the USA. 
#The goal is to analyze and understand the factors associated with crime rates and to provide insights into crime patterns across different states.

#Minimum :
    
#Minimum Murder Rate: 0.8 (North Dakota has the lowest murder rate).
#Minimum Assault Rate: 45 (North Dakota has the lowest assault rate).
#Minimum Urban Population (UrbanPop): 32 (Vermont has the smallest urban population).
#Minimum Rape Rate: 7.3 (North Dakota has the lowest rape rate).

#Maximum:

#Maximum Murder Rate: 17.4 (Georgia has the highest murder rate).
#Maximum Assault Rate: 337 (North Carolina has the highest assault rate).
#Maximum Urban Population (UrbanPop): 91 (California has the largest urban population).
#Maximum Rape Rate: 46.0 (Nevada has the highest rape rate).

#Constraints :
    
#The analysis should consider the relationships and patterns among the variables (Murder, Assault, UrbanPop, Rape) to draw meaningful conclusions.
#Statistical and data visualization techniques should be used to interpret the data and discover insights about crime rates in different states.

# imort libries

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import seaborn as sns


data = pd.read_csv("E:/datascience/clustaring/crime_data (1).csv")
data

#####################################

data.head()

data.tail()

##########################################

data.shape

#########################################

data.dtypes

#########################################

data.describe()
#############################################

# Check for missing values
data.isnull().sum()

# Select relevant features for clustering
selected_features = data[['Murder', 'Rape', 'Assault', 'UrbanPop']]

# Standardize the data


scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

#  linkage matrix for hierarchical clustering

linkage_matrix = linkage(scaled_data, method='ward')

# Plot the dendrogram

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, p=5, truncate_mode='level')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

##################################

data.columns

##################################

#Bar Plot

data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia'],
    'Murder': [13.2, 10.0, 8.1, 8.8, 9.0, 7.9, 3.3, 5.9, 15.4, 17.4]
}

# Create a DataFrame
data = pd.DataFrame(data)

# Create a bar plot for 'Murder' column
plt.figure(figsize=(10, 6))
plt.bar(data['State'], data['Murder'])
plt.title('Murder Rates by State')
plt.xlabel('State')
plt.ylabel('Murder Rate')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability

plt.show()

###################################

# Histogram

# Create subplots
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.histplot(data['Murder'], kde=True)
plt.title('Histogram of Murder Rates')
plt.xlabel('Murder Rate')

###################################

# Scatter Plot

plt.subplot(2, 2, 2)
plt.scatter(data.index, data['Murder'], label='Murder Rate', color='b', marker='o')
plt.title('Scatter Plot of Murder Rates')
plt.xlabel('State Index')
plt.ylabel('Murder Rate')

##########################################

# Probability Density Function (PDF)

plt.subplot(2, 2, 3)
sns.kdeplot(data['Murder'], shade=True, color='g')
plt.title('Probability Density Function (PDF) of Murder Rates')
plt.xlabel('Murder Rate')
plt.show()

###########################################

# Cumulative Distribution Function (CDF)
plt.subplot(2, 2, 4)
sns.kdeplot(data['Murder'], cumulative=True, color='r')
plt.title('Cumulative Distribution Function (CDF) of Murder Rates')
plt.xlabel('Murder Rate')

plt.tight_layout()

plt.show()

#########################################


