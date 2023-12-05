import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

def getProjections():
    data = pd.read_csv('Lab 1 Spreadsheet - 2011.csv', usecols=[2,3,4,5,6,7,8,9])

    attributes = data[["Number of People Migrated from New York", "Population with Different State of Residence 1 Year Ago", "Median Age Living in State (Years)", "Population 65+ Years Old (%)", "25+ Years Old Population with Bachelor's Degree or Higher (%)", "No Healthcare Coverage (%)", "Households with One or More People Under 18 Years (%)", "Civilian Employment Rate (%)"]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(attributes)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    loadings = pca.components_[:2, :]
    print(loadings)

    #plotted points
    projections_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    projections_df.to_csv('points.csv')

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

    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.show()

    # It can be determined from the resulting plot that the elbow point is at 3

    optimal_k =  3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    mds_data['cluster'] = kmeans.fit_predict(features)
    mds_data.to_csv('mds_data_clusters.csv', index=False)

    
if __name__ == "__main__":
    # getProjections()
    # findEuclidean()
    elbowMethod()