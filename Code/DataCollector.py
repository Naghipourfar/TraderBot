import json

import pandas as pd
import requests

"""
    Created by Mohsen Naghipourfar on 8/16/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

# connect to poloniex's API
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1356998100&end=9999999999&period=300'

coins = ['BTC']  # LTC, BCC, ETH, STELLAR, CARDANO
df_list = []
start_timestamp = str(1439014500)
period = str(7200)
for coin in coins:
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_' + coin + '&start=' + start_timestamp + '&end=9999999999&period=' + period
    url_content = requests.get(url)
    df = json.loads(url_content.content)
    df = pd.DataFrame(df)
    original_columns = [u'close', u'date', u'high', u'low', u'open']
    new_columns = ['Close', 'Timestamp', 'High', 'Low', 'Open']
    df = df.loc[:, original_columns]
    df.columns = new_columns
    df.to_csv('../Data/' + coin + '2015to2017.csv', index=None)
