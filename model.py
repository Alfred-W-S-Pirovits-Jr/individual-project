import os
import pandas as pd
import seaborn as sns
import env
import wrangle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.api import Holt, ExponentialSmoothing


from sklearn.metrics import mean_squared_error
from math import sqrt 


# # for presentation purposes
# import warnings
# warnings.filterwarnings("ignore")
from statsmodels.tsa.api import Holt, ExponentialSmoothing

# evaluate
from sklearn.metrics import mean_squared_error
from math import sqrt 

#Splits the dataset
def train_test_split(df, time_duration):
    flights_fortnightly_mean = df.resample(time_duration).mean()
    
    # split into train, validation, test
    train = flights_fortnightly_mean[:'2016']
    validate = flights_fortnightly_mean['2017' : '2018']
    test = flights_fortnightly_mean['2019' : ]

    return train, validate, test

#graphs the data and shows the split
def graph_split(train, validate, test):

        plt.figure(figsize=(12,4))
        plt.plot(train['average_delay'])
        plt.plot(validate['average_delay'])
        plt.plot(test['average_delay'])
        plt.ylabel('average_delay')
        plt.title('average_delay')
        plt.show()

# rmse function
def evaluate(target_var, validate, yhat_df):
    
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 2)
    return rmse

# plots the target values for train validate and predicted and plots y_hat while also showint RMSE
def plot_and_eval(target_var, train, validate, yhat_df):

    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, validate, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

# appends eval_df with rmse for tests
def append_eval_df(model_type, target_var, validate, yhat_df, eval_df):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, validate, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return pd.concat([eval_df, d])

#BASELINES

def last_average_baseline(train, validate, yhat_df, eval_df):
    # take the last average from the train set
    last_average = train['average_delay'][-1:][0]


    yhat_df = pd.DataFrame(
        {'average_delay': [last_average]},
        index=validate.index)

    yhat_df.head()

    plot_and_eval('average_delay', train, validate, yhat_df)

    eval_df = append_eval_df('last_observed_value', 
                                 'average_delay', validate, yhat_df, eval_df)
    return eval_df


def total_average_baseline(train, validate, yhat_df, eval_df):
    # get the average of fortnightly delays from the train set
    average_of_fortnightly_means = round(train['average_delay'].mean(), 2)


    yhat_df = pd.DataFrame(
        {'average_delay': [average_of_fortnightly_means]},
        index=validate.index)

    yhat_df.head()

    plot_and_eval('average_delay', train, validate, yhat_df)

    eval_df = append_eval_df('average_of_all_test_means', 
                                 'average_delay', validate, yhat_df, eval_df)
    return eval_df

def rolling_average_baselines(train, validate, yhat_df, eval_df):

    #Rolling averages for 1 fortnight, 4 weeks, 12 weeks, 26 weeks and 1 year

    periods = [1, 2, 6, 13, 26]

    for p in periods: 
        rolling_average_delay = round(train['average_delay'].rolling(p).mean()[-1], 2)

        yhat_df = pd.DataFrame({'average_delay': [rolling_average_delay]},
                                index=validate.index)

        model_type = str(p) + '_fortnight_moving_avg'

        for col in train.columns:
            eval_df = append_eval_df(f'rolling_average_of_{p}_fortnights',
                                    'average_delay', validate, yhat_df, eval_df)
    return eval_df

#returns seasonal only returns fit 3 for now as it is the best for AA but will work on later
def holts_average_delay(train, validate, yhat_df, eval_df):
    
    #Loops through all the fits
    for i in range(1,5):
    
        hst_average_delay_fit1 = ExponentialSmoothing(train.average_delay, seasonal_periods=26, trend='add', seasonal='add').fit()
        hst_average_delay_fit2 = ExponentialSmoothing(train.average_delay, seasonal_periods=26, trend='add', seasonal='mul').fit()
        hst_average_delay_fit3 = ExponentialSmoothing(train.average_delay, seasonal_periods=26, trend='add', seasonal='add', damped=True).fit()
        hst_average_delay_fit4 = ExponentialSmoothing(train.average_delay, seasonal_periods=26, trend='add', seasonal='mul', damped=True).fit()

        # hst_list = [hst_average_delay_fit1, hst_average_delay_fit2, hst_average_delay_fit3, hst_average_delay_fit4]
        # best_hst = min(hst_list)

        results_average_delay=pd.DataFrame({'model':['hst_average_delay_fit1', 'hst_average_delay_fit2', 'hst_average_dalay_fit3', 'hst_average_delay_fit4'],
                                      'SSE':[hst_average_delay_fit1.sse, hst_average_delay_fit2.sse, hst_average_delay_fit3.sse, hst_average_delay_fit4.sse]})
        results_average_delay.sort_values(by='SSE')

        yhat_df = pd.DataFrame({'average_delay': hst_average_delay_fit3.forecast(validate.shape[0] + 1)},
                                  index=validate.index)
        yhat_df.head()

        plot_and_eval('average_delay', train, validate, yhat_df)

        eval_df = append_eval_df('holts_seasonal', 'average_delay', validate, yhat_df, eval_df)

        return eval_df
    
def holt_linear(train, validate, yhat_df, eval_df):
    linear_model = Holt(train['average_delay'], exponential=False, damped=True)
    linear_model = linear_model.fit(optimized=True)
    yhat_values = linear_model.predict(start = validate.index[0],
                              end = validate.index[-1])
    yhat_df['average_delay'] = round(yhat_values, 2)

    plot_and_eval('average_delay', train, validate, yhat_df)

    eval_df = append_eval_df('holts_linear_trend', 'average_delay', validate, yhat_df, eval_df)
    return eval_df

def previous_period(train, validate, yhat_df, eval_df):
    # find previous 2 year periods and append to validate as prediction
    yhat_df = train.loc['2015':'2017'] + train.diff(53).mean()

    # yhat_df = index of validate
    yhat_df.index = validate.index

    plot_and_eval('average_delay', train, validate, yhat_df)
    
    eval_df = append_eval_df('previous 2 years', 'average_delay', validate, yhat_df, eval_df)

    return eval_df

