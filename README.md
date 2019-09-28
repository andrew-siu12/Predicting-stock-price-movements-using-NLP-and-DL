# Predicting-stock-price-movements-using-news

Using NLP and Deep learning to classify sentiment of news headline in order to predict stock price movements.

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
In real world, stocking trading involve much more than monitoring the stock price. Traders use vast amount of market information, news and reports such as SEC 8-K filings to trade stocks. In this project, we will perform multi-class text classification to predict the stock price movement by analyzing news articles and market information to classify whether stocks goes up, stay, down. For many years, traditional NLP employed machine learning models suc as SVM, logistic regression and Naive Bayes to solve problems. Words often represent using one-hot encodin which generate huge sparse vectors and lead to poblems such as the curse of dimensionality.  Theis made the learning of the model very difficult and time comsuming. Such representation also provide no semantic meaning of words. As such, distributed vector representation such as word2vec comes to help. Neural networkds attempt to learn multiple levels of representation of increasing complexity/abstraction. As of this reason, we use deep learning models along with distributed word represntation for this project.

### Data
There are no public avaliable dataset for stock news headline. We first scraped the tickers of top 1% market capitalization of NASDAQ and NYSQ stock exchange. Once the ticker is collected, we scraped six years of news and prices for each of the tickers we collected. The run time of data collection is determined by the number of tickers e.g. top N% market capitalization. It can take more than a month to collect if we set N to large percentage. Since this is a classification task, we determine the labels of news as follows: calculate the relative return of the stock over a period of time (over 7 stock trading days in this project) and label up if the relative return is greater than 1, stay if the relative return between -1 and 1 and down if the relative return is lower than -1. Finally, we concatenate the news and labels together. We collected around 58824 news in total. However, most company have multiple news in one day, we will only use single news per day for all of the company. This leave us only 6618 news in total. 

### Process 
* **Data collection** - Collect data and create news-label dataset (described in previous section)
* **Text preprocessing** - remove punctuation, stopwords and malformed words, lowercase, lemmatize and finally tokenize words 
* **Train Test Split** - Split the processed data into training and test set
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


## Improvement

## Credit
