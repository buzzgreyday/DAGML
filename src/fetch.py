from typing import List, Any

from src.classes import Transactions


async def fetch(type: str, session, url):
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            if type.lower() in ('t', 'trans', 'transact', 'transaction', 'a', 'addr', 'address'):
                return data['data']
            elif type.lower() in ('l', 'loc', 'location', 'local', 'gl', 'geo', 'geoloc', 'geolocal', 'geolocation'):
                country = data['country']['iso_code']
                provider = data['traits']['isp']
                return country, provider
            else:
                raise TypeError(f'"{type.title() if type is not None else "None"}" is not a valid parameter')


async def transaction(data, session, url) -> List:
    transaction_data = await fetch('transaction', session, url)
    if transaction_data:
        return Transactions(transaction_data).clean_merge(data)
    else:
        print(f'Could not retrieve transactional data from {url}')
        return []


async def addresses(session, url) -> tuple[Any, Any] | list[Any]:
    """
    Returns a list of validator node addresses or an empty list if failed to get validator node addreses
    :param session:
    :param url:
    :return list: validator node wallet addresses
    """
    data = await fetch('address', session, url)
    if data:
        return data
    else:
        print(f'Could not retrieve address data from {url}')
        return []