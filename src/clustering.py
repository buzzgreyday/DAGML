import pandas as pd
from sklearn.cluster import KMeans


def optimal_k(wallet_features_scaled):
    """
    Saves elbow plot to aid determine optimal number of clusters
    """
    import matplotlib.pyplot as plt

    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(
            n_clusters=k,
            n_init=100,
            max_iter=1000,
            random_state=42
        )
        kmeans.fit(wallet_features_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(20, 12))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Elbow Method for Optimal k')
    plt.savefig(f'elbow_plot.png')  # Got four clusters


async def fit_and_predict(wallet_features: pd.DataFrame, wallet_features_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Fit K-Means to number of clusters
    :return the non-scaled dataframe with a new column containing cluster number:
    """
    kmeans = KMeans(
        n_clusters=4,
        n_init=100,
        max_iter=1000,
        random_state=42
    )
    wallet_features['cluster'] = kmeans.fit_predict(wallet_features_scaled)
    return wallet_features
