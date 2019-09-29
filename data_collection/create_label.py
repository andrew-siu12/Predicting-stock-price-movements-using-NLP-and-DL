import json
import datetime
from math import log
import pandas as pd


def calc_mid_long_return(ticker, date, delta, priceSet):
    """
    Calculate the relative return of the stock over a period of time either short(1-day),
    mid(7-days) or long(28-days)
    If ticker not in s&p 500: just return the percentage change of stock price against its own
    If in s&p 500: return percentage change of stock price against the index


    :param ticker: str. ticker of company
    :param date: datetime object
    :param delta: difference in days
    :param priceSet: json file contains the price of the tickers
    :return: True, return_self_per
    """

    baseDate = datetime.datetime.strptime(date, "%Y-%m-%d")
    prevDate = (baseDate - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
    nextDate = (baseDate + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if delta == 1:
        wkday = baseDate.weekday()
        if wkday == 0:  # Monday
            prevDate = (baseDate - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        elif wkday == 4:  # Friday
            nextDate = (baseDate + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        elif wkday == 5:  # Saturday
            prevDate = (baseDate - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            nextDate = (baseDate + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        elif wkday == 6:  # Sunday
            prevDate = (baseDate - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
            nextDate = (baseDate + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        return_self = (priceSet[ticker]['adjClose'][date] - priceSet[ticker]['adjClose'][prevDate]) / \
                      priceSet[ticker]['adjClose'][prevDate]
        return_sp500 = (priceSet['^GSPC']['adjClose'][date] - priceSet['^GSPC']['adjClose'][prevDate]) / \
                       priceSet['^GSPC']['adjClose'][prevDate]
        return_self_per = round(return_self, 4) * 100
        return_sp500_per = round(return_sp500, 4) * 100

        # else:
        #     return_self = (priceSet[ticker]['adjClose'][date] - priceSet[ticker]['adjClose'][prevDate]) / \
        #                   priceSet[ticker]['adjClose'][prevDate]
        #     return_sp500 = (priceSet['^GSPC']['adjClose'][date] - priceSet['^GSPC']['adjClose'][prevDate]) / \
        #                    priceSet['^GSPC']['adjClose'][prevDate]
        #     return_self_per = round(return_self, 4) * 100
        #     return_sp500_per = round(return_sp500, 4) * 100

        return True, return_self_per - return_sp500_per  # relative return
    except Exception as e:
        return False, 0

def return_to_json():
    """
    Write short, mid, long returns of all tickers collected from tickers.py to a json file
    :return:
    """

    raw_price_file = 'inputs/stockPrices_raw.json'
    with open(raw_price_file) as file:
        print("Loading price info ...")
        priceSet = json.load(file)
        dateSet = priceSet['AAPL']['adjClose'].keys()

    returns = {'short': {}, 'mid': {}, 'long': {}}  # 1-depth dictionary
    for num, ticker in enumerate(priceSet):
        print(num, ticker)
        for term in ['short', 'mid', 'long']:
            returns[term][ticker] = {}  # 2-depth dictionary
        for day in dateSet:
            date = datetime.datetime.strptime(day, "%Y-%m-%d").strftime("%Y%m%d")
            tag_short, return_short = calc_mid_long_return(ticker, day, 1, priceSet)
            tag_mid, return_mid = calc_mid_long_return(ticker, day, 7, priceSet)
            tag_long, return_long = calc_mid_long_return(ticker, day, 28, priceSet)
            if tag_short:
                returns['short'][ticker][date] = return_short
            if tag_mid:
                returns['mid'][ticker][date] = return_mid
            if tag_long:
                returns['long'][ticker][date] = return_long

    with open('./inputs/stockReturns.json', 'w') as outfile:
        json.dump(returns, outfile, indent=4)

