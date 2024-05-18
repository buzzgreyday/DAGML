import asyncio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from aiohttp import ClientSession, TCPConnector
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler


class RequestTransactions:
    def __init__(self, session):
        self.session = session
        self.seconds_wait = 6

    async def explorer(self, url):
        async with self.session.get(url) as resp:
            if resp.status == 200:
                transaction_data = await resp.json()
                if transaction_data:
                    return transaction_data.get('data')
                else:
                    print(f'Transactional data from {url}: IS empty')
                    return
            else:
                print(f'Could not retrieve transactional data from {url}')
                await asyncio.sleep(self.seconds_wait)


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


async def fetch_transaction(session, url) -> List:
    async with session.get(url) as response:
        if response.status == 200:
            transaction_data = await response.json()
            return Transactions(transaction_data['data']).clean()
        else:
            print(f'Could not retrieve transactional data from {url}')
            return []


async def fetch_addresses(session, url) -> List:
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            return data['data']
        else:
            print(f'Could not retrieve transactional data from {url}')
            return []


async def get_addresses():
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        task = fetch_addresses(session, f'https://dyzt5u1o3ld0z.cloudfront.net/mainnet/validator-nodes')
        data = await asyncio.gather(task)
        addresses = []
        for d in data[0]:
            addresses.append(d['address'])
        return addresses


async def request_transactions(addresses) -> Tuple[List[List]]:
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        tasks = [
            fetch_transaction(session, f'https://be-mainnet.constellationnetwork.io/addresses/{address}/transactions/sent')
            for address in addresses]
        transactions = await asyncio.gather(*tasks)
        return transactions


async def create_dataframe(data: Tuple) -> pd.DataFrame:
    """
    Takes the bulk of data from a transaction object (list of lists) and returns a dataframe
    :param data:
    :return:
    """
    list_of_data = []
    for row in data:
        data = pd.DataFrame(row, columns=['source', 'destination', 'amount', 'timestamp'])
        data = data.sort_values(by=['timestamp', 'source'])
        list_of_data.append(data)
    data = pd.concat(list_of_data)
    data['timestamp'] = pd.to_datetime(data.timestamp, format='ISO8601')
    return data


async def destination_specific_calculations(data) -> pd.DataFrame:
    """
    Calculate the sum sent to specific wallet addresses (destinations)
    :param data:
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
                    on=['source', 'destination', 'total_amount_sent_to_destination'], how='left')
    return data


async def handle_outliers(data, cutoff_transaction_count, cutoff_date) -> pd.DataFrame:
    """
    Clean data and handle radical outliers
    :param data:
    :param cutoff_transaction_count:
    :param cutoff_date:
    :return:
    """
    # Ignore non-frequent destinations
    data = data[data['count'] >= cutoff_transaction_count]
    # Control timespan
    data = data[data['timestamp'] >= cutoff_date]
    # Exclude individual outliers
    data = data[~(data['source'] == 'DAG2AhT8r7JoQb8fJNEKFLNEkaRSxjNmZ6Bbnqmb')]
    return data


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

    wallet_features = transactions.groupby('source').agg(
        total_amount_sent=pd.NamedAgg(column='amount', aggfunc='sum'),
        number_of_transactions=pd.NamedAgg(column='amount', aggfunc='count'),
        average_amount_sent=pd.NamedAgg(column='amount', aggfunc='mean'),
        # Low transaction frequency means infrequent engagement or usage
        transaction_frequency=pd.NamedAgg(column='timestamp', aggfunc=lambda x: x.diff().mean().total_seconds())
    ).dropna().reset_index()

    # Normalize values for ML

    scaler = StandardScaler()
    wallet_features_scaled = scaler.fit_transform(wallet_features.drop(columns='source'))
    print(wallet_features_scaled)

    # Determine optimal number of clusters
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

    # Fit K-Means to number of clusters
    kmeans = KMeans(
        n_clusters=4,
        n_init=100,
        max_iter=1000,
        random_state=42
    )
    wallet_features['cluster'] = kmeans.fit_predict(wallet_features_scaled)

    # Analyse what clusters represent
    cluster_analysis = wallet_features.groupby('cluster').agg(
        total_amount_sent=pd.NamedAgg(column='total_amount_sent', aggfunc='sum'),
        average_amount_sent=pd.NamedAgg(column='average_amount_sent', aggfunc='mean'),
        wallet_count=pd.NamedAgg(column='source', aggfunc='count'),
        transaction_frequency=pd.NamedAgg(column='transaction_frequency', aggfunc='mean')
    ).reset_index()
    print(cluster_analysis)

    # Custom formatter function to avoid scientific notation
    def currency_formatter(x, pos):
        return '{:,.2f}'.format(x)

    # Visualization: Scatter plot of clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='transaction_frequency',
        y='average_amount_sent',
        hue='cluster',
        size='total_amount_sent',
        palette='viridis',
        data=wallet_features,
        s=100,
        alpha=0.6
    )
    plt.title('Cluster Visualization Based on Total Tokens Sent and Average Transaction Value')
    plt.xlabel('Transaction Freq')
    plt.ylabel('Average Amount Sent')
    # Applying custom formatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.legend(title='Cluster')
    plt.savefig('cluster_scatterplot.png')

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
    cutoff_date = pd.to_datetime('2022-05-17').tz_localize('UTC')
    cutoff_transaction_count = 8
    await define_clusters(cutoff_transaction_count, cutoff_date)
    exit(0)



if __name__ == '__main__':
    asyncio.run(main())