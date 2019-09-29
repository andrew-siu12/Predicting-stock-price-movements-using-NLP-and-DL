from data_collection.tickers import get_tickers
from data_collection.scrape_news import  *
from data_collection.create_label import *
from data_collection.stock_price import *
import sys


def main():

    if len(sys.argv) < 2:
        print('Usage: data_collection/all_tickers.py <int_percent>')
        return
    top_n = sys.argv[1]
    get_tickers(int(top_n))  # keep the top N% market-cap companies

    print("Sucessfully retrived all the tickers")

    get_stock_prices()
    print("Sucessfulyy retrived all the price data")

    # it may takes weeks to get all the relevant data
    reuter_crawler = ReutersCrawler()
    reuter_crawler.run()

    print("Sucessfully retrived all the news")

    return_to_json()

    print("Sucessfully retrived return of all tickers")


if __name__ == "__main__":
    main()