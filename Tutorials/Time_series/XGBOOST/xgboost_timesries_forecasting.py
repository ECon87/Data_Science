"""
Based on the ML Master tutorial of the same name.
URL: https://machinelearningmastery.com/xgboost-for-time-series-forecasting/


- What is XGBoost?
XGBoost is short for Extreme Gradient Boosting and it is an efficient implementation of stochastic gradient boosting for classification and regression problems.
It is fast and efficient, and oftens offers the best performance on a wide range of predictive modeling tasks (winner in multiple Kaggle competitions).
XGBoost is an ensemble of decision trees and provides access to a suite of model hyperparameters designed to provide control over the model training process.
"The most important factor behind the success of XGBoost is its scalability in all scenarios" XGBoost: A Scalable Tree Boosting Systme, 2016

- Can XGBoost be used in time series forecasting?
Yes, but we first need to transform the time series databaset into a supervised learning problem.
XGB is designed for classification and regression on tabular datasets.

Requires the use of a specialized technique for evaluating the model called "walk-forward validation".
We need to do this since k-fold cross validation results in optimistically biased results.
For example, it's not valid to fit the model on the data from the future and have it predict the futre.
This is the key fact that makes k-fold inapplicable.
With walk-forward validation  -
    * The dataset is first split into train and test sets by selecting a cut point;
    * for example, all data except the last 12 days is used for training and the last 12 days is used for testing.
    * If we are interested in making a one-step forecast (e.g., one month), then we can evaluate the model by training on the training dataset and predicting the first step in the test dataset.
        We can then add the real observation from the test set to the training dataset, then have the model predict the second step in the test dataset.
    * Repeating this process for the entire test dataset will give a one-step prediction for the entire test dataset from which an error measure can be calculated to evaluate the skill of the model.
    * Read -
        https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/



Goal - Develop an XGBoost model for time series forecastin
Learnings:
    - XGBoost is an implementation of the gradient boosting ensemble algorithm for classification and reression
    - Time series datatest can be transformed into supervised leanring using a sliding-window representation
    - How to fit, evaluate, and make predictions with an XGBoost model for time series forecasting.
"""

# load libraries
import pandas as pd
from numpy import asarray
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import xgboost

print(xgboost.__version__)

# XGBoost has its own python api, but we can use XGBoost using the scikit-learn API
# via the XGBRegression wrapper class

# Time series data preparation
# Given a seq of numbers, we will restructure to look like a supervised learning model
# To achieve this, we can use previous time steps as input variables and use the next time step
# as the output variable (i.e., predict next time-step from previous time-steps)

def series_to_supervised(data, n_in=1, n_out = 1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, .., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # Put it all together
    agg = pd.concat(cols, axis=1)
    return agg

def series_to_supervised(data, n_in=1, n_out = 1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, .., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # Put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows wiht NaN values
    if dropnan:
        agg = agg.dropna()
    return agg.values


# Define a walk-forward validation function
# Takes the entire supervised learning version of the time series dataset and
# the number of rows to use as the test as arguments.
# It the steps through the test set, calling the xgboost_forecast() function to
# make a one-step forecast. An error measure is calculated and the details are returned for analyis.
# Walk-forward validation for univariate data


def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        textX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, textX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions


def train_test_split(data, n_test):
    """Splits the dataset while respecting its temporal nature. Works for univariate time series"""
    return data[:-n_test, :], data[-n_test:, :]


# Define the xgboost_forecast fucntio that implements the algorithm;
# the inputs are the training dataset and test input row, it fits a model and makes
# a one-step prediction


def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and ouptut columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = xgboost.XGBRegressor(objective='reg:squarederror',
                                 n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# =============================================================
# Apply to a real world problem - daily total femable births
# Forecast montly birth rates with xgboost
# =============================================================

DIR = '/home/econ87/Documents/Data_Science/Tutorials/Time_series/Data/xgboost_time_series/'

# load dataset
series = pd.read_csv(f'{DIR}/daily-total-female-births.csv',
                     header=0,
                     index_col=0)

values = series.values


# plot series - no obvious trends or seasonality
series.plot()
plt.show()

# transform the time series data into suprvised learning
data = series_to_supervised(values, n_in = 6)

# evaluate
mae, y, yhat = walk_forward_validation(data, 12)
print(f'MAE: {mae:.3f}')

preds_plots_df = series.iloc[-12:].copy()
preds_plots_df['Predicted'] = yhat
preds_plots_df.plot()
plt.show()


plt.plot(y, label = 'Expected')
plt.plot(yhat, label = 'Predicted')
plt.xticks(range(12), labels = series.index[-12:].values)
plt.legend()
plt.show()


# ========================
# Out-of-sample forecasts
# ========================
# Once a final XGBoost model configuration is chosen, a model can be finalized and used to make a predediction on new data.
# This is called an out-of-sample forecast. This is identical to making a prediction
# during the evaluation of the model: as we always want to evaluate a model using the same
# procedure that we expect to use when the model is used to make prediction on new data.

# The example below demonstrates fitting a final XGBoost model on all available data and
# making a one-step prediction beyond the end of the dataset.

# Note that we don't need the walk_forward_validation function now

# load dataset
series = pd.read_csv(f'{DIR}/daily-total-female-births.csv',
                     header=0,
                     index_col=0)
values = series.values

# transform the time series data in to supervised learning
train = series_to_supervised(values, n_in = 6)

# split into input and output columns
trainX, trainy = train[:, :-1], train[:, -1]

# fit model
model = xgboost.XGBRegressor(objective = 'reg:squarederror', n_estimators = 1000)
model.fit(trainX, trainy)

# construct an input for a new prediction
row = values[-6:].flatten()

# make a one-step prediction
yhat = model.predict(asarray([row]))
print('Input: %s. Predicted: %.3f' %(row, yhat[0]))
