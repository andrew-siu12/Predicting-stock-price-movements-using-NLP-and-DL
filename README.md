# Predicting-stock-price-movements-using-news

Using NLP and Deep learning to predict stock price movements.

#### -- Project Status: [Active]

### Technologies
* Python
* Pandas, numpy
* BeautifulSoup
* spacy
* keras

## Project Description

In the internet age, the interest in online news websites have grown rapidly and it allows publisher to produce hundreds of news per day.  This is especially the case for financial market since it changes every seconds. This provides a huge amount of unstructured data that can be analyzed to incorporate into investors decision making process. However, finding and monitoring news on the web and distilling the information contained in them remains a formidable task because of the proliferation of diverse sites.  Each site typically contains a huge volume of text that is not always easily deciphered.  Market participants  will have difficulty extracting and summarizing these information in quick and efficient way. Given the increase in amount of data, deep learning approaches to financial modeling has gained a lot of attention from practitioners.

Stock price prediction is very challenging problem due to the amount of unforseen events that cause stock price to change. 
In real world, stocking trading involve much more than monitoring the stock price. Traders use vast amount of market information, news and reports such as SEC 8-K filings to trade stocks. In this project, we will perform multi-class text classification to predict the stock price movement by analyzing news articles and market information to classify whether stocks goes up, stay, down. For many years, traditional NLP employed machine learning models suc as SVM, logistic regression and Naive Bayes to solve problems. Words often represent using one-hot encoding which generate huge sparse vectors and lead to poblems such as the curse of dimensionality.  Theis made the learning of the model very difficult and time comsuming. Such representation also provide no semantic meaning of words. As such, distributed vector representation such as word2vec comes to help. Neural networks attempt to learn multiple levels of representation of increasing complexity/abstraction. As of this reason, we use deep learning models along with distributed word represntation for this project.

### Data
There are no public avaliable dataset for stock news headline. We first scraped the tickers of top 5% market capitalization of NASDAQ, NYSQ and AMSE stock exchange. Once the ticker is collected, we scraped five and half years of news and prices for each of the tickers we collected. The run time of data collection is determined by the number of tickers e.g. top N% market capitalization. It can take more than a month to collect if we set N to large percentage. For each news, one month , one week change of price were calculation and normalized by the change in the S&P 500 index.  Since this is a classification task, we determine the labels of news as follows: calculate the relative daily return of the stock before and after news release and  normalized by the change in S&P 500 index for the same time period. Then we label "up" if the relative return is greater than 1, "stay" if the relative return between -1 and 1 and "down" if the relative return is lower than -1. Finally, we concatenate the news and labels together. We collected around 160000 news in total. However, most company have multiple news in one day, we will only use single news per day for all of the company. This leave us only around 25000 news in total. The dataset consist of the following features:
* **VIX (volatility index)** - represents the market's expectation of 30-day forward-looking volatility.
* **monthly/weekly stock movement** - percentage change of the stock price normalized by the change in S&P500 over the same period of time
* **combined_tokens** - corpus of news headlines and first sentence of news with stopwords and punctuations removed, lemmatized and transform into pre-trained fastText word embeddings.

### Process 
* **Data collection** - Collect data and create news-label dataset (described in previous section)
* **Text preprocessing** - remove punctuation, stopwords and malformed words, lowercase, lemmatize and finally tokenize words 
* **Train Test Split** - Randomly shuffled and split the processed data into 80% training and 20% test set.
* **Create Model for training** - we defined three models: Convolutional Neural Network (CNN) for text classification inspired by [Yoon Kim](https://arxiv.org/abs/1408.5882)(2014), Bidirectional LSTM and Bidirectional LSTM with attention. 
* **Hyerparameter Tuning** - used Hyperas (Bayesion optimization package) for hyperparameter tuning.
* **Evaluate performance** - used the tuned models to predict the test set and compare the performance of three models using accuracy and F1 for metrics.

## File Descriptions
```
.
├── README.md
├── base
│   ├── base_model.py
│   └── base_train.py
├── config
│   ├── Bilstm.json
│   ├── att_bilstm.json
│   └── cnn.json
├── data_collection
│   ├── create_label.py
│   ├── scrape_news.py
│   ├── stock_price.py
│   └── tickers.py
├── data_loader
│   └── data_generator.py
├── main.py
├── models
│   ├── attention_Bilstm.py
│   ├── cnn.py
│   └── lstm.py
├── notebook
│   ├── Create_label.ipynb
│   ├── Data\ Preparation.ipynb
│   ├── Hyperparameter-Tuning.ipynb
│   ├── stock\ data\ collection.ipynb
│   └── tickers.ipynb
├── saved_models
│   ├── lstm_model_fast_text.h5
│   ├── tuned_attBilstm_fast_text.h5
│   └── tuned_cnn_model_fast_text.h5
├── trainers
│   └── model_trainer.py
└── utils
    ├── __init__.py
    ├── config.py
    └── util.py

```
1. `base` folder contains the base class for model building and training.
2. `config` folder contains the configuration of model building and training parameters (json file).
3. `data_collection` -  Run `generate_data.py` to collect price and news and create labels. Run `generate_data.py` 
4. `data_loader` - Run `data_generator.py` to generate training and testing data
5. `models` - contains three different models architecture. 
6. `notebook` - collection of jupyter notebook of initial prototyping and hyperparameter tuning of models
7. `saved_models` - tuned models weight saved in h5 format
8. `trainers` - class for training the models defined in `models` folder
9. `utils` - collection of utility scripts
10. `main.py`- The script contains the whole procedure of training data generation, model buildin, training and evaluate the performance of model. Run `python main.py config/cnn.json` to run CNN model.

## Results
All three models were trained for 50 epochs and batch size of 32 with early stopping. We used bayesian optimisation to tune the hyperparameters. All three models achieved similar results after tuning. CNN achieved best loss and f1 score out of the three models. BiLSTM with self-attention has the highest accuracy. However, the tuning and training time of CNN is the fastest. We benchmarked these models against all random choice baseline model of 33.3%. The result of all three models show improvement of 95%.  These results suggests that combining word-embeddings and market information and the use of neural networks can achieve a good result. 

| Model     | Loss | Accuracy| f1 score| 
| -------------   |:--------:| --------:| -------:|
| CNN| 0.822622| 0.645083 |0.664479|
| BiLSTM | 0.825816| 0.64086| 0.661145| 
| BiLSTM with self-attention | 0.825047| 0.645887| 0.664052 |


<p float="left">
  <img src="https://i.imgur.com/xZD5RKy.png" width="500" /> 
  <img src="https://i.imgur.com/dzYF0Bj.png" width="500" />
  <img src="https://i.imgur.com/HuE7jtm.png" width="500" />
</p>

## Improvement

(1) The definition of labels is very different from traditional text classification. We used the relative stock price changes on the day where the news release. The mdoels are able to predict the price movement fairly well as suggested by the f1 score. However, we should use more rigirous approach to define the label. As future work, we need to get the exact time of the news release and use the one-minute price before news release to calculate the daily return. Moreover, in order to observe the full impact on the news. We could use relative return on the stock over 5 trading days period as our label. 

(2) Another imporvement concerns with the number of features of data. There are only three market information features and one news feature to use in modelling which limit our predictive power. Also, this may be the reason that we get similar results on all three models after hyperparameter tuning. Some market features include log daily return, log weekly return, log monthly return, daily market residualized return etc. For news, we may include category of news, volume counts for each news etc. Also, we need more effective way to collect data. We consider downloading data to SQL database

(3) The third improvement consideration includes using newer NLP techniques such as ELMo, ULMFit and BERT for text classification.
