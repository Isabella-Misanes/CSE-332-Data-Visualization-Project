import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

def findEuclidean():
    data = pd.read_csv('Lab 1 Spreadsheet - 2011.csv', usecols=[2,3,4,5,6,7,8,9])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    distance_matrix = pairwise_distances(scaled_data, metric='euclidean')
    
    mds = MDS(n_components=2, dissimilarity='precomputed')

    embedded_data = mds.fit_transform(distance_matrix)
    df_embedded = pd.DataFrame(data=embedded_data, columns=['X', 'Y'])
    df_embedded.to_csv('mds_data.csv', index=False)

def elbowMethod():
    mds_data = pd.read_csv('mds_data.csv')

    features = mds_data[['X', 'Y']]

    k_values = range(1, 11)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)

    # It can be determined from the resulting plot that the elbow point is at 3

    optimal_k =  3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    mds_data['cluster'] = kmeans.fit_predict(features)
    mds_data.to_csv('mds_data_clusters.csv', index=False)

    
if __name__ == "__main__":
    #findEuclidean()
    #elbowMethod()
    pass