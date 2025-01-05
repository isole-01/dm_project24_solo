
import pandas
import copy

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.model_selection import ParameterSampler
import pandas
import numpy as np
import seaborn
import sys

sys.path.append("src")

from transformations import center_and_scale

RANDOM_STATE = 13



import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

def remove_outliers_dbscan(df, eps=0.5, min_samples=5):
    """
    Removes rows from a DataFrame identified as outliers using DBSCAN, with normalization.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - eps (float): The maximum distance between two samples for them to be considered neighbors.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
    - pd.DataFrame: A DataFrame with outliers removed.
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['number'])

    # Normalize the numerical columns
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(numerical_cols)

    # Apply DBSCAN to detect outliers
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(normalized_data)

    # Keep rows that are not outliers (-1 indicates an outlier in DBSCAN)
    filtered_df = df[clusters != -1]

    return filtered_df

def remove_outliers_zscore(df, threshold=3):
    """
    Removes rows from a DataFrame where any numerical column has a z-score greater than the given threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): The z-score threshold to identify outliers (default is 3).

    Returns:
    - pd.DataFrame: A DataFrame with outliers removed.
    """
    # Compute z-scores for all numerical columns
    z_scores = df.select_dtypes(include=['number']).apply(zscore)

    # Keep rows where all z-scores are within the threshold
    filtered_df = df[(z_scores.abs() <= threshold).all(axis=1)]

    return filtered_df

def c_models(dataset):

    numeric_dataset = dataset.select_dtypes(include="number")

    normalized_df, normalization_scalers_df = center_and_scale(numeric_dataset)

    
    # computing sample densities for DBScan to avoid arbitrary values
    # maximum distance between any two points in the dataset
    maximum_distance = abs(normalized_df.max() - normalized_df.min()).sum().item()
    average_concentration = normalized_df.shape[0] / maximum_distance
    
    # print(normalized_df.shape, normalized_df.values )
    # set model space
    base_algorithms = [
        KMeans,
        SpectralClustering,
        DBSCAN,
        AgglomerativeClustering,
        OPTICS
    ]
    base_algorithms_names = [
        "kmeans",
        "spectral",
        "dbscan",
        "agglomerative",
        "optics"
    ]
    # what hyperparameters does each model have?
    hyperparameters_per_algorithm = [
        {  # KMeans
            "n_clusters": list(range(3, 7)) ,
            "max_iter": [500],
            "n_init": [10, 20],
            "init": ["k-means++", "random"],
            "random_state": [RANDOM_STATE]
        },
        {  # SpectralClustering
            "n_clusters": list(range(3, 7)) ,
            "affinity": ["rbf", "nearest_neighbors"],
            "eigen_tol": [1e-3, 1e-4],
            "random_state": [RANDOM_STATE]
        },
        {  # DBSCAN
            "eps": average_concentration * np.array([10, 5, 2.5, 1, 0.1, 0.01, 0.0001]),
            "min_samples": [5, 10, 15],
            "metric": ["euclidean", "manhattan"]
        },
        {  # AgglomerativeClustering
            "n_clusters": list(range(3, 7)) ,
            "linkage": ["ward", "complete", "average","single"],
            "metric": ["euclidean"]
        },
        {
            "p": list(range(1, 10, 2))
        }
    ]
    
    results_per_algorithm = list()
    fit_models = list()
    clusterings = list()
    for algorithm, algorithm_name, hyperparameters in zip(base_algorithms, base_algorithms_names, hyperparameters_per_algorithm):
        # setup search for this algorithm
        sampled_hyperparameters = list(ParameterSampler(
            copy.deepcopy(hyperparameters),
            n_iter=5,  # how many configurations to sample?
            random_state=RANDOM_STATE
        ))
        models = [
            algorithm(**selected_hyperparameters).fit(normalized_df.values)
            for selected_hyperparameters in sampled_hyperparameters
        ]
        clusterings += [
            model.labels_
            for model in models
        ]
        fit_models += models
    
        # store configurations
        for fit_model, selected_hyperparameters in zip(models, sampled_hyperparameters):
            selected_hyperparameters["algorithm"] = algorithm_name
            if hasattr(fit_model, "n_clusters"):
                selected_hyperparameters["n_clusters"] = fit_model.n_clusters
            else:
                selected_hyperparameters["n_clusters"] = selected_hyperparameters.get("n_clusters", 0)
        results_per_algorithm += sampled_hyperparameters
    
    results_df = pandas.DataFrame.from_records(results_per_algorithm)
    results_df.loc[:, "random_state"] = RANDOM_STATE
    
    # clean extra hyperparameters
    results_df.loc[results_df["max_iter"].isna(), "max_iter"] = -1
    results_df.loc[results_df["eps"].isna(), "eps"] = -1
    results_df.loc[results_df["p"].isna(), "p"] = -1
    
    results_df = results_df.astype({"n_clusters": int, "random_state": int, "p": int})
    results_df


    silhouette_per_model = [
    silhouette_score(normalized_df, clustering) if len(set(clustering)) > 1 else -1
    for clustering in clusterings
]
    results_df.loc[:, "silhouette"] = silhouette_per_model
    results_df = results_df.sort_values(by="silhouette", ascending=False)

    from scipy.spatial.distance import cdist


    clusterings_cohesions = list()
    for i, clustering in enumerate(clusterings):
        clusters_indices = [np.argwhere(clustering == target_cluster).squeeze()
                            for target_cluster in np.unique(clustering)]
        centroids = [normalized_df.values[cluster].mean(axis=0)
                     for cluster in clusters_indices]
        mean_distances = [sum(cdist([centroid], normalized_df.values[indices]).squeeze()).item()
                          for centroid, indices in zip(centroids, clusters_indices) if centroid.ndim > 0]
        clusterings_cohesions.append(mean_distances)
    
    summed_cohesions = [sum(cohesion) for cohesion in clusterings_cohesions]
    results_df.loc[:, "cohesion"] = summed_cohesions
    results_df.sort_values(by="silhouette")

    return results_df, clusterings, fit_models



def stage_sort_key(stage):
    # Handle special cases
    if stage == 'prologue':
        return (-1, 0)  # Comes before all other stages
    if stage == 'result':
        return (float('inf'), 0)  # Single race event, placed last but labeled as stage 1

    # Handle standard stage formats
    parts = stage.split('-')
    if len(parts) > 1:
        try:
            # Extract the numeric part
            base_stage = int(parts[1].rstrip('abcdefghijklmnopqrstuvwxyz'))
            # Extract the letter part (if present) and assign a secondary sort order
            sub_stage = ord(parts[1][-1]) - ord('a') + 1 if parts[1][-1].isalpha() else 0
            return (base_stage, sub_stage)
        except ValueError:
            return (float('inf'), 0)  # Handle unexpected formats gracefully
    
    return (float('inf'), 0)  # For invalid stage formats

# Function to assign stage numbers and total stages
def stage_number(df):
    # Preserve the original index
    original_index = df.index

    # Create a DataFrame with unique stages
    unique_stages = (
        df[['stage']].drop_duplicates()  # Only consider unique stages within the group
        .assign(sort_key=lambda x: x['stage'].apply(stage_sort_key))  # Add sorting keys
        .sort_values(by='sort_key')  # Sort by the sort key
        .reset_index(drop=True)  # Reset index after sorting
    )
    
    # Assign stage numbers
    unique_stages['stage_number'] = range(1, len(unique_stages) + 1)
    
    # Calculate total stages
    unique_stages['total_stages'] = unique_stages['stage_number'].max()
    
    # Drop the temporary sort_key column
    unique_stages = unique_stages.drop(columns=['sort_key'])
    
    # Merge the stage numbers back to the original DataFrame
    result = df.merge(unique_stages, on='stage', how='left')

    # Restore the original index
    result.index = original_index
    return result
