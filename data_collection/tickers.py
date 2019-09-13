import csv
from urllib.request import urlopen
import numpy as np


def get_tickers(percent):
    """
    Get the tickers of top percent market cap and write it in the csv file
    :param percent: int 0-100
    :return:
    """

    assert isinstance(percent, int)
    file = open('./inputs/tickerList.csv', 'w')
    writer = csv.writer(file, delimiter=',')
    cap_stat, output = np.array([]), []
    for exchange in ["NASDAQ", "NYSE"]:
        url = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange="
        repeat_times = 10 # repeat downloading in case of http error
        for _ in range(repeat_times):
            try:
                print("Downloading tickers from {}...".format(exchange))
                response = urlopen(url + exchange + '&render=download')
                content = response.read().decode('utf-8').split('\n')
                for num, line in enumerate(content):
                    line = line.strip().strip('"').split('","')
                    if num == 0 or len(line) != 9:
                        continue # filter unmatched format
                    # ticker, name, last_sale, market_cap, IPO_year, sector, industry
                    ticker, name, _, market_cap, _, _, _ = line[0:4] + line[5:8]
                    cap_stat = np.append(cap_stat, float(market_cap))
                    output.append([ticker, name.replace(',', '').replace('.', ''),
                                   exchange, market_cap])
                break
            except:
                continue

    for data in output:
        market_cap = float(data[3])
        if market_cap < np.percentile(cap_stat, 100 - percent):
            continue
        writer.writerow(data)
