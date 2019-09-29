import sys
import re
import os
import time
import random
import json
import yfinance as yf


def get_stock_prices():
    fin = open('./inputs/tickers.txt')
    output = './inputs/stockPrices_raw.json'

    # exit if the output already existed
    if os.path.isfile(output):
        sys.exit("Prices data already existed!")

    price_set = {}
    for num, line in enumerate(fin):
        ticker = line.strip()
        print(num, ticker)
        price_set[ticker] = repeat_download(ticker)

    with open(output, 'w') as outfile:
        json.dump(price_set, outfile, indent=4)


def repeat_download(ticker, start_date='2011-07-06', end_date='2017-02-21'):
    repeat_times = 2  # repeat download for N times
    for i in range(repeat_times):
        try:
            time.sleep(random.uniform(2, 5))
            price_str = get_price_from_yahoo(ticker, start_date, end_date)
            if price_str:  # skip loop if data is not empty
                return price_str
        except Exception as e:
            print(e)
            if i == 0:
                print(ticker, "Http error!")


def get_price_from_yahoo(ticker, start_date, end_date):
    quote = yf.download(ticker, start_date, end_date)

    quote.columns = ['open', 'high', 'low', 'close', 'adjClose', 'volume']
    quote.index = quote.index.strftime('%Y-%m-%d')

    quote_dict = quote.to_dict()

    return quote_dict

