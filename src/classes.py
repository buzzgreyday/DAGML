from typing import List


class Transactions:

    def __init__(self, transactions):
        self.transactions = transactions

    def clean_merge(self, data: List[List]):
        """
        0(n2) time
        :param data:
        :return:
        """
        results = []
        for transaction in self.transactions:
            for d in data:
                if d[0] == transaction['source']:
                    results.append(
                        [
                            str(transaction['source']),
                            str(transaction['destination']),
                            int(transaction['amount']) / 100000000,
                            str(transaction['timestamp']),
                            str(d[1]),
                            str(d[2]),
                            str(d[3])
                        ]
                    )
                    break
        return results