import asyncio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from aiohttp import ClientSession, TCPConnector
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler


class Transactions:

    def __init__(self, transactions):
        self.transactions = transactions

    def clean(self):
        results = []
        for transaction in self.transactions:
            results.append(
                [
                    str(transaction['source']),
                    str(transaction['destination']),
                    int(transaction['amount']) / 100000000,
                    str(transaction['timestamp'])
                ]
            )
        return results


async def fetch(session, url):
    async with session.get(url) as response:
        if response.status == 200:
            transaction_data = await response.json()
            return transaction_data['data']


async def fetch_transaction(session, url) -> List:
    transaction_data = await fetch(session, url)
    if transaction_data:
        return Transactions(transaction_data).clean()
    else:
        print(f'Could not retrieve transactional data from {url}')
        return []


async def fetch_addresses(session, url) -> List:
    """
    Returns a list of validator node addresses or an empty list if failed to get validator node addreses
    :param session:
    :param url:
    :return validator node wallet addresses:
    """
    data = await fetch(session, url)
    if data:
        return data
    else:
        print(f'Could not retrieve address data from {url}')
        return []


async def get_addresses() -> List:
    """
    Request the current list of validators from API and return a list of validator node wallet addresses
    :return validator node wallet addresses:
    """
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        task = fetch_addresses(
            session,
            f'https://dyzt5u1o3ld0z.cloudfront.net/mainnet/validator-nodes'
        )
        data = await task
        addresses = []
        for d in data:
            addresses.append(d['address'])
        return addresses


async def request_transactions(addresses) -> Tuple[List[List]]:
    """
    Creates a list of lists containing transactional data for wallet addresses.
    Each list in the list is a transaction.
    :param addresses:
    :return:
    """
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        tasks = [
            fetch_transaction(
                session,
                f'https://be-mainnet.constellationnetwork.io/addresses/{address}/transactions/sent'
            )
            for address in addresses]
        transactions = await asyncio.gather(*tasks)
        return transactions


async def create_dataframe(data: Tuple[List[List]]) -> pd.DataFrame:
    """
    Takes the bulk of data from a transaction object (list of lists) and returns a dataframe
    :param data:
    :return:
    """
    list_of_data = []
    for row in data:
        data = pd.DataFrame(row, columns=['source', 'destination', 'amount', 'timestamp'])
        data = data.sort_values(by=['timestamp', 'source'])
        if not data.empty:
            list_of_data.append(data)
    data = pd.concat(list_of_data)
    data['timestamp'] = pd.to_datetime(data.timestamp, format='ISO8601')
    return data


async def destination_specific_calculations(data) -> pd.DataFrame:
    """
    Calculate the sum sent to specific wallet addresses (destinations) and count how many transactions has been sent
    to specific wallets historically. This data can be used to filter out addresses with for example less than 5
    transactions (if you're looking to identify an exchange wallet).
    :param dataframe with added address specific data:
    :return:
    """
    data.loc[:, 'amount_sent_to_destination_sum'] = (
        data.groupby(['source', 'destination'])['amount'].transform('sum'))
    # Group by 'source', 'destination', and 'total_amount_sent_to_destination', count occurrences, and sort by count
    # to get number of transactions to specific addresses
    count_transactions_to_destination = data.groupby(
        ['source', 'destination', 'amount_sent_to_destination_sum']).size().reset_index(
        name='count'
    ).sort_values(
        by='count'
    )

    # Merge the counts back to the original DataFrame
    data = pd.merge(data, count_transactions_to_destination,
                    on=['source', 'destination', 'amount_sent_to_destination_sum'], how='left')
    return data


async def handle_outliers(data, cutoff_transaction_count, cutoff_date) -> pd.DataFrame:
    """
    Clean data and handle radical outliers
    :param data:
    :param cutoff_transaction_count:
    :param cutoff_date:
    :return clean dataframe with no outliers:
    """
    # Ignore non-frequent destinations
    data = data[data['count'] >= cutoff_transaction_count]
    # Control timespan
    data = data[data['timestamp'] >= cutoff_date]
    # Exclude individual outliers
    data = data[~(data['source'] == 'DAG2AhT8r7JoQb8fJNEKFLNEkaRSxjNmZ6Bbnqmb')]
    return data


async def create_wallet_features(data: pd.DataFrame) -> pd.DataFrame.groupby:
    """
    Self-explanatory
    :param transactional data:
    :return features for ML:
    """
    wallet_features = data.groupby('source').agg(
        total_amount_sent=pd.NamedAgg(column='amount', aggfunc='sum'),
        number_of_transactions=pd.NamedAgg(column='amount', aggfunc='count'),
        average_amount_sent=pd.NamedAgg(column='amount', aggfunc='mean'),
        # Low transaction frequency means infrequent engagement or usage
        transaction_frequency=pd.NamedAgg(column='timestamp', aggfunc=lambda x: x.diff().mean().total_seconds())
    ).reset_index().copy()
    # Check for any NaN values in transaction_frequency and fill them with a suitable value, e.g., the mean or median
    wallet_features['transaction_frequency'].fillna(0)
    return wallet_features


async def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize values for ML
    :return scaled wallet features:
    """
    scaler = StandardScaler()
    wallet_features_scaled = scaler.fit_transform(data.drop(columns='source'))
    return wallet_features_scaled


def optimal_k(wallet_features_scaled):
    """
    Saves elbow plot to aid determine optimal number of clusters
    """
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


async def generate_cluster_analysis(wallet_features: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse what clusters represent
    :param wallet_features:
    :return:
    """
    cluster_analysis = wallet_features.groupby('cluster').agg(
        total_amount_sent=pd.NamedAgg(column='total_amount_sent', aggfunc='sum'),
        average_amount_sent=pd.NamedAgg(column='average_amount_sent', aggfunc='mean'),
        wallet_count=pd.NamedAgg(column='source', aggfunc='count'),
        transaction_frequency=pd.NamedAgg(column='transaction_frequency', aggfunc='mean')
    ).reset_index()
    print(cluster_analysis)
    return cluster_analysis


def visualize_clusters_barplot(cluster_analysis):
    # Visualization: Bar plot of total tokens sent by cluster
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='cluster',
        y='average_amount_sent',
        data=cluster_analysis,
        legend=False
    )
    plt.title('Total amount Sent by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Amount Sent')
    plt.savefig('cluster_bar_plot.png')


def visualize_clusters_scatterplot(wallet_features):
    """
    Visualize clusters in a scatterplot
    :param wallet_features:
    :return:
    """
    def currency_formatter(x, pos):
        return '{:,.2f}'.format(x)

    # Visualization: Scatter plot of clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='total_amount_sent',
        y='average_amount_sent',
        hue='cluster',
        size='transaction_frequency',
        palette='viridis',
        data=wallet_features,
        s=100,
        alpha=0.6
    )
    plt.title('Cluster Visualization Based on Total Tokens Sent and Average Transaction Value')
    plt.xlabel('Total Amount Sent')
    plt.ylabel('Average Amount Sent')
    # Applying custom formatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.legend(title='Cluster')
    plt.savefig('cluster_scatterplot.png')



async def define_clusters(cutoff_transaction_count, cutoff_date):
    """
    Categorize node wallet sell/transaction behavior
    :param cutoff_transaction_count:
    :param cutoff_date:
    :return:
    """
    pd.set_option('display.float_format', '{:.2f}'.format)
    addresses = await get_addresses()
    transactions = await request_transactions(addresses)
    transactions = await create_dataframe(transactions)
    transactions = await destination_specific_calculations(transactions)
    transactions = await handle_outliers(transactions, cutoff_transaction_count, cutoff_date)
    wallet_features = await create_wallet_features(transactions)
    wallet_features_scaled = await scale_features(wallet_features)
    optimal_k(wallet_features_scaled)
    wallet_features = await fit_and_predict(wallet_features, wallet_features_scaled)
    cluster_analysis = await generate_cluster_analysis(wallet_features)
    visualize_clusters_scatterplot(wallet_features)
    visualize_clusters_barplot(cluster_analysis)


    # Step 8: Determine which wallets are in which clusters
    wallet_clusters = wallet_features[
        [
            'source',
            'cluster',
            'average_amount_sent',
            'total_amount_sent',
            'transaction_frequency',
            'number_of_transactions'
        ]
    ]
    print(wallet_clusters)
    # Example: List wallets in each cluster
    for cluster in wallet_clusters['cluster'].unique():
        print(f"\nWallets in Cluster {cluster}:")
        print(wallet_clusters[wallet_clusters['cluster'] == cluster]['source'].values)
    print(wallet_clusters[(wallet_clusters.source == 'DAG3nt2qnhdeGS5ZxredSxqaZrn9KoL6xHZ3yTc5')])


async def main():
    """Initiate processes"""
    # Specify the cutoff date
    cutoff_date = pd.to_datetime('2022-12-31').tz_localize('UTC')
    cutoff_transaction_count = 8
    await define_clusters(cutoff_transaction_count, cutoff_date)
    exit(0)



if __name__ == '__main__':
    asyncio.run(main())