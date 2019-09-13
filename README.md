# Predicting-stock-price-movements-using-news

Using recent advancement in NLP to classify sentiment of news headline in order to predict stock price movements.

#### -- Project Status: [Active]

### Technologies
* Python
* Pandas, numpy
* BeautifulSoup
* spacy
* keras

## Project Description

In the internet age, the interest in online news websites have grown rapidly and it allows publisher to produce hundreds of news per day.  This is especially the case for financial market since it changes every seconds. This provides a huge amount of unstructured data that can be analyzed to incorporate into investors decision making process. However, finding and monitoring opinion sites on the Web and distilling the information contained in them remains a formidable task because of the proliferation of diverse sites.  Each site typically contains a huge volume of opinion text that is not always easily deciphered.  The average human reader will have difficulty extracting and summarizing the opinions in sites. 

In this project, we will perform multi-class sentiment analysis to predict the stock price movement by analyzing news articles and tries to classify news as postive, neutral and negative (up, stay, down in pricing sense). Traditionally, the approach to perform sentiment analysis is by using rule-based method.  Data mining involves extract information from a data set and transform it intoan understandable structure.  However, using data mining to extract useful features andselect the best of those is very challenging to undertake in high amount of data settings.For this reason, we are going to adopt Deep learning to determine the price of stocks andthe overall market based on financial news data.


### Data
There are no public avaliable dataset for stock news headline. We first scraped the tickers of top 1% market capitalization of NASDAQ and NYSQ stock exchange. Once the ticker is collected, we scraped six years of news and prices for each of the tickers we collected. The run time of data collection is determined by the number of tickers e.g. top N% market capitalization. It can take more than a month to collect if we set N to large percentage. Since this is a classification task, we determine the labels of news as follows: calculate the relative return of the stock over a period of time (over 7 stock trading days in this project) and label up if the relative return is greater than 1, stay if the relative return between -1 and 1 and down if the relative return is lower than -1. Finally, we concatenate the news and labels together. We collected around 58824 news in total. However, most company have multiple news in one day, we will only use single news per day for all of the company. This leave us only 6618 news in total. 

### Process 
* **Data collection** - Collect data and create news-label dataset (described in previous section)
* **Text preprocessing** - remove punctuation, stopwords and malformed words, lowercase, lemmatize and finally tokenize words 

## File Descriptions
```
├── base
│   ├── base_model.py
│   └── base_train.py
├── config
│   └── mlp.json
├── data_collection
│   ├── create_label.py
│   ├── generate_data.py
│   ├── scrape_news.py
│   ├── stock_price.py
│   └── tickers.py
├── data_loader
│   └── data_generator.py
├── inputs
│   ├── constituents.csv
│   ├── news_failed_tickers.csv
│   ├── news_price_df.csv
│   ├── news_reuters.csv
│   ├── stockPrices_raw.json
│   ├── stockReturns.json
│   ├── tickerList.csv
│   └── tickers.txt
├── models
│   └── mlp.py
├── notebook
│   ├── Data\ Preparation.ipynb
│   ├── Modelling.ipynb
│   ├── stock\ data\ collection.ipynb
│   └── tickers.ipynb
└── utils
    ├── __init__.py
    ├── config.py
    └── util.py
```
1. `base` folder contains the base class for model building and training.
2. `config` folder contains the configuration of model building and training parameters (json file).
3. `data_collection` -  Run `generate_data.py` to collect price and news and create labels. Run `generate_data.py` 
4. `data_loader` - Run `data_generator.py` to generate training and testing data
5. `inputs` - folder contains all the collected data
6. `models` - 
7. `notebook` - collection of jupyter notebook of initial prototyping.
8. `utils` - collection of utility scripts

## Summary
