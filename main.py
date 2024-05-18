import asyncio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

from aiohttp import ClientSession, TCPConnector
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter

from sklearn.preprocessing import StandardScaler


# addresses = [
#     "DAG3nt2qnhdeGS5ZxredSxqaZrn9KoL6xHZ3yTc5",
#     "DAG4Ewq5dFeG2YyCzqCooG6gSn6js5YmzD9w3cn4",
#     "DAG4GiYssahypSmykMp4ALwzQMZqNg35Sibx1TUJ",
#     "DAG4KWAEjg7ARXgJKEghyrz5Y8c5Fm6iqMThYfPu",
#     "DAG4LUKibZHggi4uhyT5CG1XHCU3EGSRXCVwxNvu",
#     "DAG5aWVLVN8SkxSvJTjjCFA7VNRcduopLcRWL6X2",
#     "DAG5byyjxGFJbKttkv14aMfmbUuAuKqSDx9sBGRV",
#     "DAG5DpN4cLaZjSKhEJ4gmpib7mRKSGueQFxFM8VF",
#     "DAG5PEPQj9AffZJZJTzLbEYWxgH2Yt82keoJXQY2",
#     "DAG6ANcW9KFoehdvuezW7s2fbjpcwTL4JjYXk2Hb"
# ]


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


async def fetch_transactions(session, url) -> List:
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

async def request_addresses():
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
            fetch_transactions(session, f'https://be-mainnet.constellationnetwork.io/addresses/{address}/transactions/sent')
            for address in addresses]
        transactions = await asyncio.gather(*tasks)
        return transactions


async def create_dataframe(data):
    print(data)
    list_of_data = []
    for row in data:
        data = pd.DataFrame(row, columns=['source', 'destination', 'amount', 'timestamp'])
        data = data.sort_values(by=['timestamp', 'source'])
        list_of_data.append(data)
    data = pd.concat(list_of_data)
    data['timestamp'] = pd.to_datetime(data.timestamp, format='ISO8601')
    return data


async def main():
    """Initiate process"""
    pd.set_option('display.float_format', '{:.2f}'.format)
    # Specify the cutoff date
    cutoff_date = pd.to_datetime('2022-05-17').tz_localize('UTC')
    cutoff_min_transaction_count = 8
    addresses = await request_addresses()
    transactions = await request_transactions(addresses)
    transactions = await create_dataframe(transactions)
    transactions.loc[:, 'total_amount_sent_to_destination'] = (
        transactions.groupby(['source', 'destination'])['amount'].transform('sum'))
    # Group by 'source', 'destination', and 'total_amount_sent_to_destination', count occurrences, and sort by count
    # to get number of transactions to specific addresses
    transaction_count_to_destination = transactions.groupby(
        ['source', 'destination', 'total_amount_sent_to_destination']).size().reset_index(
        name='count'
        ).sort_values(
        by='count'
    )

    # Merge the counts back to the original DataFrame
    transactions = pd.merge(transactions, transaction_count_to_destination,
                            on=['source', 'destination', 'total_amount_sent_to_destination'], how='left')

    """ Handle radical outliers and clean """
    # Ignore non-frequent destinations
    transactions = transactions[transactions['count'] >= cutoff_min_transaction_count]
    # Control timespan
    transactions = transactions[transactions['timestamp'] >= cutoff_date]
    # Remove individual outliers
    transactions = transactions[~(transactions['source'] == 'DAG2AhT8r7JoQb8fJNEKFLNEkaRSxjNmZ6Bbnqmb')]


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
    plt.savefig(f'elbow_plot.png') # Got four clusters

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
    exit(0)


if __name__ == '__main__':
    asyncio.run(main())