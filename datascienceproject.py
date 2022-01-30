#!/usr/bin/env python
# coding: utf-8

# In[250]:


import pandas as pd
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics




snp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
ta_url = "https://en.wikipedia.org/wiki/TA-125_Index"
# api_key = "b87d58d0d348ad7cf92e6293b53d7720"


# In[252]:


def load_soup_object(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    return soup


# In[3]:


def get_snp_stocks_symbols(url):
    soup = load_soup_object(url)
    stocks = soup.find_all('tbody')[0]
    stocks_names = list()
    stocks_symbols = list()
    for row in stocks('tr'):
        cels = row("a")
        stocks_symbols.append(cels[0].get_text().strip())
        stocks_names.append(cels[1].get_text().strip())
    stocks_symbols.remove('Symbol')
    stocks_names.remove('SEC filings')
    df = pd.DataFrame({"stock_name":stocks_names,"stock_symbol":stocks_symbols})
    return df
    
    
    


# In[4]:


def get_nasdaq_stocks_symbols(url):
    soup = load_soup_object(url)
    stocks = soup.find_all('tbody')[3]
    stocks_names = list()
    stocks_symbols = list()
    for row in stocks('tr'):
        names = row("a")
        stocks_names.append(names[0].get_text().strip())
        stocks_symbols.append(row.get_text().splitlines()[2])
    
    stocks_names.remove("GICS")
    stocks_symbols.remove("Ticker")
    df = pd.DataFrame({"stock_name":stocks_names,"stock_symbol":stocks_symbols})
    return df


# In[5]:


def get_ta_stocks_symbols(url):
    soup = load_soup_object(url)
    stocks = soup.find_all('tbody')[1]
    stocks_names = list()
    stocks_symbols = list()
    for row in stocks('tr'):
        name = row("a")
        if(len(name)>0):
            stocks_names.append(name[0].text)
        else:
            stocks_names.append(row.get_text().splitlines()[1])
            
        stocks_symbols.append(row.get_text().splitlines()[3])
        
    stocks_names.remove("Name")
    stocks_symbols.remove("Symbol")
    df = pd.DataFrame({"stock_name":stocks_names,"stock_symbol":stocks_symbols})
    return df


# 

# In[6]:


def concat_tabels(snp_df,nasdaq_df,ta_df):
    combined_df = pd.concat([snp_df, nasdaq_df,ta_df],ignore_index = True, axis=0)
    return combined_df


# In[7]:


def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


# In[8]:


# def get_stocks_prices(stocks):
#     keys = stocks['stock_symbol'].tolist()
#     data_dict = {}
#     index = 0
#     dates_full = []
#     for stock in stocks.iterrows():
#         symbol = stock[1].at['stock_symbol']
#         price_list = []
#         dates = []
#         data = yf.Ticker(symbol)
#         prices = data.history(start="2020-01-01", end="2022-01-01",interval ="1mo")
#         for i in range(len(prices)):
#             price_list.append(prices['Close'][i])
#             dates.append(str(prices['Close'].keys()[i].date()))
#         if(len(dates) > len(dates_full)):
#             dates_full = dates
#         data_dict[keys[index]] = price_list
#         index+=1 
#     df = pd.DataFrame.from_dict(data_dict, orient='index',
#                        columns=dates_full)
#     return df
        
        


# In[9]:


def get_stocks_data(stocks,start_date,end_date,intervals):
    keys = stocks['stock_symbol'].tolist()
    data = yf.download(keys,start=start_date, end=end_date,interval =intervals)
    df = pd.DataFrame(dict(data['Close'].items()))
    return df


# In[10]:


def remove_missing_rows(df,x):
    if x==0:
        df2=df.dropna(axis=0, how="any", subset=None, inplace=False)
    else:    
        df2=df.dropna(axis=0, how="any", thresh=x, subset=None, inplace=False)
    return df2


def remove_missing_cols(df,x):
    if x==0:
        df2=df.dropna(axis=1, how="any", subset=None, inplace=False)
    else:    
        df2=df.dropna(axis=1, how="any", thresh=x, subset=None, inplace=False)
    return df2


# In[11]:


def get_experts_recomendations(page_url):
    url = "https://www.wallstreetzen.com/stock-screener/stock-forecast" + page_url
    soup = load_soup_object(url)
    data = soup.find('tbody', attrs = {"class" : "MuiTableBody-root-504"})
    tickers = []
    predictions = []
    for row in data('tr'):
        tickers.append(row.find('a').text)
        predictions.append(row.find_all('td')[7].text)
    df = pd.DataFrame(predictions,index = tickers, columns = ['prediction'])
    return df



# In[12]:


def create_recomendations_data(start,prediction_df):
    symbols = prediction_df.index.values.tolist()
    data =  yf.download(symbols[start:start + 100],'2009-01-13','2022-01-24','1mo')
    df = pd.DataFrame(dict(data['Close'].items()))
    return df


# In[13]:


def create_recomendations_df():
    df = get_experts_recomendations("")
    for i in range(2,10):
        page_url = "?t=6&p="+str(i)+"&s=mc&sd=desc"
        df = pd.concat([df,get_experts_recomendations(page_url)],ignore_index = False, axis=0)
    return df


# In[19]:


def get_recomendation_data(recomendations_df):
    
    recomendations_data  = pd.DataFrame()
    for start in range(0,801,100):
        recomendations_data = pd.concat([recomendations_data,create_recomendations_data(start,recomendations_df)],ignore_index = False, axis=1)
    return recomendations_data


    


# In[131]:


def get_increase_decrease_data(symbol):
    snp_stock = yf.download([symbol],start = "1984-01-01",end = "2021-01-01", interval = "1mo")
    df = pd.DataFrame(snp_stock)
    return df


# In[21]:


snp_symbols_df = get_snp_stocks_symbols(snp_url)
nasdaq_symbols_df = get_nasdaq_stocks_symbols(nasdaq_url)
ta_symbols_df = get_ta_stocks_symbols(ta_url)
stocks_symbols_df = remove_duplicates(concat_tabels(snp_symbols_df,nasdaq_symbols_df,ta_symbols_df))
stocks_symbols_df


# In[22]:


stocks_data_df = get_stocks_data(stocks_symbols_df,'2007-01-01','2022-01-01','1mo')
stocks_data_df


# In[23]:


stocks_data_df = remove_missing_rows(stocks_data_df,1)
stocks_data_df = remove_missing_cols(stocks_data_df,0)
stocks_data_df


# In[256]:


def imaging_change_in_prices(df):
    increase_dict = dict()
    df = df.transpose()
    for row in df.iterrows():
        for i in range(5,len(row[1]),6):
            increase_dict.setdefault(i, []).append(1 if row[1][i] > row[1][0] else 0)

    prepare_data_for_graph(df,increase_dict)
    
def prepare_data_for_graph(df,increase_dict):
    amount_increase_list = []
    amount_decrease_list = []
    months_list = []
    num_of_increase = 0
    num_of_decrease = 0
    for column in increase_dict:
        for val in increase_dict[column]:
            if val == 1:
                num_of_increase+=1 
            else:
                num_of_decrease +=1
        
        amount_increase_list.append(num_of_increase)
        amount_decrease_list.append(num_of_decrease)
        months_list.append(column)
        num_of_increase = 0
        num_of_decrease = 0
        
        d = {'increase' : amount_increase_list ,'decrease':amount_decrease_list }
    graph_df = pd.DataFrame(d,index = months_list)
    show_graph(graph_df)
    
def show_graph(graph_df):
    ax = graph_df.plot.bar(rot=0)
    ax.set_xticklabels([t if not i%2 else "" for i,t in enumerate(ax.get_xticklabels())])

        
imaging_change_in_prices(stocks_data_df)    


# In[ ]:





# In[257]:


def predict_stock_price(symbol,stocks_prices_df):
    stock_prices = stocks_prices_df[symbol]
    stock_prices = pd.DataFrame(stock_prices)
    stock_prices
    stock_prices['days_from_start'] = (stock_prices.index - stock_prices.index[0]).days;
    print(stock_prices)
    X_train, X_test, y_train, y_test = train_test_split(stock_prices.days_from_start,stock_prices[symbol])
    plt.scatter(X_train,y_train,label='Training Data' , color= 'r', alpha = .7)
    plt.scatter(X_test,y_test,label='Testing Data' , color= 'g', alpha = .7)
    plt.legend()
    plt.title("train test split")
    plt.show()
    LR = LinearRegression()
    LR.fit(X_train.values.reshape(-1,1),y_train)

    prediction = LR.predict(X_test.values.reshape(-1,1))
    plt.plot(X_test,prediction,label = 'Linear Regression', color = 'b', alpha = .7)
    plt.scatter(X_test,y_test,label = 'Actual test data',color= 'g', alpha = .7)
    plt.legend()
    plt.show()


# In[258]:


predict_stock_price('AAPL',stocks_data_df)


# In[28]:


recomendations_df = create_recomendations_df()


# In[29]:


recomendation_data = get_recomendation_data(recomendations_df)


# In[30]:



recomendation_data= remove_missing_cols(recomendation_data,3280)
recomendation_data = remove_missing_rows(recomendation_data,0)


# In[31]:


def remove_missing_recomendation(prediction_df,df2):
    for row in prediction_df.iterrows():
        if row[0] not in df2.columns:
            prediction_df = prediction_df.drop(row[0])  
    return prediction_df


# In[39]:


def get_avg_monthly_change(recomendation_data):
    print(recomendation_data)
    monthly_change = 0.0
    change_rates = []
    for row in recomendation_data.iterrows():
        for i in range(1,len(row)):
            diff = row[1][i] - row[1][i-1]
            val = row[1][i-1]
            monthly_change += (diff * 100) / (val)
        monthly_change = monthly_change/len(row)
        change_rates.append(monthly_change)
    return change_rates


# In[40]:


get_avg_monthly_change(recomendation_data)


# In[34]:


recomendations_df_copy = recomendations_df.copy()
recomendations_df_copy = remove_missing_recomendation(recomendations_df_copy,recomendation_data)
recomendations_df_copy


# In[50]:


recomendation_data = recomendation_data.transpose()


# In[52]:



change_rates = get_avg_monthly_change(recomendation_data)
change_rates
train_dataset = pd.DataFrame(zip(change_rates,recomendation_data.iloc[:,-1:].values.flatten(),recomendations_df_copy['prediction']),
                             index = recomendations_df_copy.index, columns = ["change", 'last_price', 'label'])
train_dataset


# In[53]:


df_max_scaled = train_dataset.copy()
df_max_scaled['last_price'] = df_max_scaled['last_price'] /df_max_scaled['last_price'].abs().max()
df_max_scaled['change'] = df_max_scaled['change'] /df_max_scaled['change'].abs().max()
df_max_scaled.plot(kind = 'scatter' , x = 'change', y = 'last_price')
plt.show()
df_max_scaled


# In[54]:


def train_model(k):
    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    label_encoded=le.fit_transform(df_max_scaled['label'])
    X = df_max_scaled.drop('label',axis = 1)
    y = df_max_scaled['label']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k)

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
    evaluate_optimal_k(X_train,y_train,X_test,y_test)
    
    
    return metrics.accuracy_score(y_test, y_pred)


# In[55]:


def evaluate_optimal_k(X_train,y_train,X_test,y_test):
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')


# In[62]:


print(train_model(10))


# In[132]:


snp_price_data = get_increase_decrease_data("^GSPC")
snp_price_data


# In[133]:


def get_increase_list(df):
    next_month_close = []
    for i in range(1,len(df)):
        next_month_close.append(0 if df['Close'][i] < df['Close'][i-1] else 1 )
    df.drop(index=df.index[0], 
            axis=0, 
            inplace=True)
    return next_month_close


# In[134]:


next_month_close =get_increase_list(snp_price_data) 
snp_price_data.insert(6,"increased", next_month_close, True)
snp_price_data


# In[272]:


snp_price_data =pd.get_dummies(snp_price_data)
snp_price_data



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(snp_price_data.drop('increased', axis =1), snp_price_data['increased'])


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[271]:


log_reg.score(X_test,y_test)


# In[249]:


snp_price_data


# In[ ]:





# In[ ]:




