import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.title('APP FOR CLUSTERING DEMAND')

st.markdown("""
This app performs demand clustering with 3 Algorithm (KMeans, AgglomerativeClustering, & DBSCAN)
* **Python libraries:** streamlit, pandas, numpy, matplotlib, sklearn
* **Data source:** Coordinates of Hydrogen Refueling Station Alternative
""")
# # user input features
# st.header('USER INPUT FEATURES')

# sidebar

st.subheader('KMeans  & AgglomerativeClustering Algorithm')
selected_range = st.slider("K range", 2, 20)

# st.subheader('AgglomerativeClustering Algorithm')
# # selected_cluster = st.number_input("Optimal Cluster", 1, 20)
# st.write(f'Optimal Cluster selected: {selected_cluster}')

st.subheader('DBSCAN Algorithm')
selected_epsilon = st.number_input("Epsilon Neighborhood", value=0.06)
selected_min_samples = st.number_input("Minimum Samples", value=1)


# File uploader for dataset
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_excel(uploaded_file)

    # Display data
    st.header('Display Coordinates Data of Hydrogen Refueling Station Alternative')
    st.write('Data Dimension: ' +
             str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    st.dataframe(data)
else:
    st.warning("Please upload an Excel file to proceed.")


# Transform latitude & longitude
def transform_points(data):
    latitude = data['Latitude']
    longitude = data['Longitude']
    demand_coordinates_transformed = np.vstack(
        [latitude, longitude]).T  # T for transpose
    # x3, y3 = transformer.transform(coordinates_transformed[:, 1], coordinates_transformed[:, 0])
    scaler = StandardScaler()
    demand_coordinat_transform_scaled = scaler.fit_transform(
        demand_coordinates_transformed)
    return demand_coordinat_transform_scaled, demand_coordinates_transformed

def perform_kmeans(coordinat_transformed, k, num_iterations=10):
    best_silhouette_score = -1
    best_kmeans = None

    for i in range(num_iterations):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        max_iter=300, n_init=10, random_state=42)
        kmeans.fit(coordinat_transformed)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(
            coordinat_transformed, cluster_labels)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_kmeans = kmeans

    return best_kmeans, best_silhouette_score

#  DBSCAN function

# Button to trigger calculation
if st.button('Calculate'):
    coordinat_transformed_scaled, coordinat_transformed = transform_points(
        data)
    range_n_clusters = list(range(2, selected_range + 1))
    best_overall_kmeans = None
    best_overall_silhouette_score = -1
    best_k = 0

    # KMeans Clustering Silhouette Scores
    st.subheader("KMeans Silhouette Scores")
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(coordinat_transformed_scaled)
        kmeans_labels = kmeans.labels_

        if len(set(kmeans_labels)) > 1:
            silhouette_avg = silhouette_score(
                coordinat_transformed_scaled, kmeans_labels)
            st.write(
                f"Number of clusters: {num_clusters}, Silhouette score: {silhouette_avg}")

            if silhouette_avg > best_overall_silhouette_score:
                best_overall_silhouette_score = silhouette_avg
                best_overall_kmeans = kmeans
                best_k = num_clusters
        else:
            st.write(
                f"Number of clusters: {num_clusters}, Cannot compute Silhouette Score with only one cluster.")

    st.subheader("Agglomerative Clustering Silhouette Scores")
    for num_clusters in range_n_clusters:
        agg_clustering = AgglomerativeClustering(
            n_clusters=num_clusters, linkage='ward')
        agg_labels = agg_clustering.fit_predict(coordinat_transformed)

        if len(set(agg_labels)) > 1:
            silhouette_agg = silhouette_score(
                coordinat_transformed, agg_labels)
            st.write(
                f"Number of clusters: {num_clusters}, Silhouette score: {silhouette_agg}")
        else:
            st.write(
                f"Number of clusters: {num_clusters}, Cannot compute Silhouette Score with only one cluster.")

    st.subheader("DBSCAN Silhouette Scores")
    dbscan = DBSCAN(eps=selected_epsilon,
                    min_samples=int(selected_min_samples))
    dbscan_labels = dbscan.fit_predict(coordinat_transformed)

    # Count number of clusters (excluding noise points)
    unique_labels = set(dbscan_labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    if num_clusters > 1:
        silhouette_dbscan = silhouette_score(
            coordinat_transformed, dbscan_labels)
        st.write(
            f"Number of clusters: {num_clusters}, DBSCAN Silhouette Score: {silhouette_dbscan}")
    else:
        st.write(
            "Cannot compute Silhouette Score with only one cluster")

    st.subheader("Best Silhoutte Score")
    st.write(
        f"The optimal number of clusters is {best_k} with a silhouette score of {best_overall_silhouette_score}")

    # Perform K-means with the optimal number of clusters
    best_cluster_labels = best_overall_kmeans.labels_
    # Add the cluster labels to the original data
    data['Cluster'] = best_cluster_labels
    # Save the clustered data to a new Excel file
    st.dataframe(data)
    fig, ax = plt.subplots()
    ax.scatter(coordinat_transformed_scaled[:, 0],
               coordinat_transformed_scaled[:, 1], c=best_cluster_labels, cmap='viridis')
    ax.scatter(best_overall_kmeans.cluster_centers_[
               :, 0], best_overall_kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
    ax.set_title('Clustered Data with Optimal K')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    st.pyplot(fig)
