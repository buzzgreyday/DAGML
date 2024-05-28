import asyncio
import os

import pandas as pd

from aiohttp import ClientSession, TCPConnector
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from src import fetch, clustering


async def get_addresses() -> List:
    """
    Request the current list of validators from API and return a list of validator node wallet addresses
    :return list: validator node wallet addresses
    """
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        task = fetch.addresses(
            session,
            f'https://dyzt5u1o3ld0z.cloudfront.net/mainnet/validator-nodes'
        )
        data = await task
        nodes = []
        for node in data:
            nodes.append([node['address'], node['ip']])
        return nodes


async def get_transactions(data: List[List]) -> Tuple[List[List]]:
    """
    Creates a list of lists containing transactional data for wallet addresses.
    Each list in the list is a transaction.
    :param data: list of nodes [[address, ip, ...], ...]
    :return:
    """
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        tasks = [
            fetch.transaction(
                data,
                session,
                f'https://be-mainnet.constellationnetwork.io/addresses/{d[0]}/transactions/sent'
            )
            for d in data]
        transactions = await asyncio.gather(*tasks)
        print(transactions)
        return transactions


async def create_dataframe(data: Tuple[List[List]]) -> pd.DataFrame:
    """
    Takes the bulk of data from a transaction object (list of lists) and returns a dataframe
    :param data: list of lists
    :return:
    """
    list_of_data = []
    for row in data:
        data = pd.DataFrame(row, columns=['source', 'destination', 'amount', 'timestamp', 'ip', 'country', 'isp'])
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
    :param data: dataframe with added address specific data
    :return pd.DataFrame:
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
    :return pd.DataFrame: clean dataframe with no outliers
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
    :param data: transactional data
    :return pd.DataFrame: grouped features for ML
    """
    wallet_features = data.groupby('source').agg(
        total_amount_sent=pd.NamedAgg(column='amount', aggfunc='sum'),
        number_of_transactions=pd.NamedAgg(column='amount', aggfunc='count'),
        average_amount_sent=pd.NamedAgg(column='amount', aggfunc='mean'),
        # Low transaction frequency means infrequent engagement or usage
        transaction_frequency=pd.NamedAgg(column='timestamp', aggfunc=lambda x: x.diff().mean().total_seconds())
    ).reset_index().copy()
    # Check for any NaN values in transaction_frequency and fill them with a suitable value, e.g., the mean or median
    wallet_features['transaction_frequency'].fillna(0, inplace=True)
    return wallet_features


async def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize values for ML
    :return scaled wallet features:
    """
    scaler = StandardScaler()
    wallet_features_scaled = scaler.fit_transform(data.drop(columns='source'))
    return wallet_features_scaled


async def label_clusters(data):
    """
    This function assigns labels based on cluster values
    :param cluster analysis dataframe:
    :return:
    """
    # lower frequency means more frequent transactions
    # Average amount sent and transaction frequency is very co-related; some (0) have a high average amount,
    # not very high transaction frequency (they doesn't sell often) but the same total amount sent as the
    # less frequent senders (1).
    # (2) A very small percentage sends very frequently and very little.
    # (3) Most sell frequently and very little
    pass


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
    # cluster_analysis = await label_clusters(cluster_analysis)
    print(cluster_analysis)
    return cluster_analysis


async def get_location(data: List[List]):
    """
    Function to get IP geolocation
    :param data: [[Addr, IP], ...]
    :return:
    """
    from dotenv import load_dotenv
    load_dotenv()
    async with ClientSession(connector=TCPConnector(ssl=False)) as session:
        for d in data:
            url = f'https://api.findip.net/{d[1]}/?token={os.getenv('GLOC_TOKEN')}'
            d.extend(await fetch('location', session, url))
            print(d)
        return data


async def define_clusters(data):
    """
    Categorize node wallet sell/transaction behavior
    :param data:
    :return:
    """
    wallet_features = await create_wallet_features(data)
    wallet_features_scaled = await scale_features(wallet_features)
    clustering.optimal_k(wallet_features_scaled)
    wallet_features = await clustering.fit_and_predict(wallet_features, wallet_features_scaled)
    cluster_analysis = await generate_cluster_analysis(wallet_features)
    visualize_clusters_scatterplot(wallet_features)
    visualize_clusters_barplot(cluster_analysis)

    # Determine which wallets are in which clusters
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
    # print(wallet_clusters)
    # Example: List wallets in each cluster
    for cluster in wallet_clusters['cluster'].unique():
        print(f"\nWallets in Cluster {cluster}:")
        print(wallet_clusters[wallet_clusters['cluster'] == cluster]["source"].values)
    print(wallet_clusters[(wallet_clusters.source == 'DAG3nt2qnhdeGS5ZxredSxqaZrn9KoL6xHZ3yTc5')])
    return wallet_clusters, cluster_analysis


async def collect_data(cutoff_transaction_count, cutoff_date):
    """
    Collect all the data needed to do the ML.
    We need to also collect price data and calculate fluctuation before transactions
    :param cutoff_transaction_count:
    :param cutoff_date:
    :return:
    """
    pd.set_option('display.float_format', '{:.2f}'.format)
    nodes = await get_addresses()
    nodes = await get_location(nodes)
    transactions = await get_transactions(nodes)
    transactions = await create_dataframe(transactions)
    transactions = await destination_specific_calculations(transactions)
    transactions_no_outliers = await handle_outliers(transactions, cutoff_transaction_count, cutoff_date)
    return transactions_no_outliers


async def main():
    """Initiate processes"""
    # Specify the cutoff date
    cutoff_date = pd.to_datetime('2022-12-31').tz_localize('UTC')
    cutoff_transaction_count = 8
    transactions_no_outliers = await collect_data(cutoff_transaction_count, cutoff_date)
    wallet_clusters, cluster_analysis = await define_clusters(transactions_no_outliers)
    exit(0)


if __name__ == '__main__':
    asyncio.run(main())
