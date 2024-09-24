
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and normalize the dataset
data = pd.read_csv("HW3/market_ds.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Finding the optimal number of clusters using the Elbow Method
inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42).fit(scaled_data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal Cluster Number')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (WCSS)')
plt.show()

# Based on the elbow method, let's assume the optimal cluster number is 4 (this can be adjusted after visualization)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Visualizing the relationship between income and spending
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    clustered_data = data[data['Cluster'] == cluster]
    plt.scatter(clustered_data['Income'], clustered_data['Spending'], label=f'Cluster {cluster}')
plt.title('Income vs Spending by Cluster')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.show()

# Visualizing the relationship between income and age
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    clustered_data = data[data['Cluster'] == cluster]
    plt.scatter(clustered_data['Income'], clustered_data['Age'], label=f'Cluster {cluster}')
plt.title('Income vs Age by Cluster')
plt.xlabel('Income')
plt.ylabel('Age')
plt.legend()
plt.show()

# At this point, based on the visualizations, you can assign names to the clusters based on their characteristics (e.g., "Young Low Spenders", "High Earners", etc.).
