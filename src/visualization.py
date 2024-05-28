from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns



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