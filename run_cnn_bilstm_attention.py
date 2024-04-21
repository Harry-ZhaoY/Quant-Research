import numpy as np
import pandas as pd
import functools 
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Bidirectional, LSTM, Attention, Concatenate, Layer
from keras.models import Model
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping
from models import cnn_bilstm_attention

# Function to fetch market data
def fetch_data(ticker_symbol, start_date, end_date):
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(interval='1h', start=start_date, end=end_date)[cols]
    new_cols = [s + "_" + ticker_symbol for s in cols]
    df.rename(columns=dict(zip(cols, new_cols)), inplace=True)
    return df
    
# Function to normalize data and create sequences
def prepare_data(df, sequence_length, y_ind):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df)
    
    xs, ys = [], []
    for i in range(len(scaled_features) - sequence_length):
        x = scaled_features[i:(i + sequence_length)]
        y = scaled_features[i + sequence_length, y_ind]  # Assuming 'Close' is at index 3
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys), scaler


# Function to create sequences
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        xs.append(data[i:(i + sequence_length)])
        ys.append(data[i + sequence_length, 3])  # Close price index
    return np.array(xs), np.array(ys)

# Perform time series cross-validation
def time_series_cross_validation(model, X, y, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold = 0
    for train_index, test_index in tscv.split(X):
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Reshape data for CNN
        X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        
        # Train your model
        print(f"Training fold {fold}...")
        model.fit([X_train_cnn, X_train], y_train, epochs=100, batch_size=32, verbose=0)  # Set verbose to 0 to avoid too much output
        
        # Evaluate your model
        test_loss = model.evaluate([X_test_cnn, X_test], y_test, verbose=0)
        print(f"Fold {fold} Test Loss: {test_loss}")

def train_test_split(X, y, train_size=0.8):
    split = int(len(X) * train_size)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

def reshape_for_cnn(X_train, X_test):
    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    return X_train_cnn, X_test_cnn

def train_model(model, X_train_cnn, X_train, y_train, X_test_cnn, X_test, y_test):
    model.compile(optimizer='adam',
              loss='mean_squared_error',  
              metrics=['mean_absolute_error', 'mean_squared_error'])  

    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        [X_train_cnn, X_train],  # Inputs for the CNN and LSTM
        y_train,  # Training targets
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X_test_cnn, X_test], y_test),  # Validation data
        callbacks=[early_stopping]
    )
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def inverse_transform(scaler, X_test, predictions, y_test):
    predictions_inverse = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], predictions), axis=1))[:, -1]
    y_test_inverse = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
    return predictions_inverse, y_test_inverse

def plot_results(y_test_inverse, predictions_inverse, portfolio_values, buy_hold_values, signals):
    # Create a figure with two subplots, one above the other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 14))  # figsize(width, height)

    # Plot the actual and predicted prices with trade signals on the first subplot
    ax1.plot(y_test_inverse, label='Actual Prices')
    ax1.plot(predictions_inverse, label='Predicted Prices')
    buy_signals = np.where(np.array(signals) == 1)[0]
    sell_signals = np.where(np.array(signals) == -1)[0]
    ax1.scatter(buy_signals, y_test_inverse[buy_signals], label='Buy Signals', marker='^', color='g')
    ax1.scatter(sell_signals, y_test_inverse[sell_signals], label='Sell Signals', marker='v', color='r')
    ax1.set_title('Actual and Predicted Prices with Trade Signals')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot the portfolio values over time on the second subplot
    ax2.plot(portfolio_values, label='Model-Based Trading Strategy')
    ax2.plot(buy_hold_values, label='Buy & Hold Strategy', linestyle='--')
    ax2.set_title('Portfolio Value Over Time: Trading Strategy vs. Buy & Hold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Portfolio Value in USD')
    ax2.axhline(y=INITIAL_INVESTMENT, color='r', linestyle=':', label='Initial Investment')
    ax2.legend()

    # Adjust the layout so that titles and labels do not overlap
    plt.tight_layout()

    # Show the figure
    plt.show()


# Trading signal
def generate_signals(predictions, actual_prices, threshold=0.02):
    """
    Generate buy and sell signals based on the model's predictions.
    
    :param predictions: Model's predicted prices.
    :param actual_prices: Actual closing prices.
    :param threshold: Minimum percentage change between the predicted and current price to trigger a signal.
    :return: Signals (1 for buy, -1 for sell, 0 for hold)
    """
    signals = [0]
    for predicted, actual in zip(predictions[1:], predictions[:-1]):  # Exclude the last actual price since there's no prediction for the next day
        change_percentage = (predicted - actual) / actual

        if change_percentage > threshold:
            signals.append(1)  # Buy
        elif change_percentage < -threshold:
            signals.append(-1)  # Sell
        else:
            signals.append(0)  # Hold
    
    return signals

# Main trading and buy-hold simulation function
def simulate_trading(signals, actual_prices, initial_investment):
    # Trading Strategy
    cash = initial_investment
    stock_quantity = 0
    portfolio_values = []

    # Buy and Hold Strategy
    buy_hold_value = initial_investment / actual_prices[0] * actual_prices  # Initial stock quantity * price at each time

    for signal, price in zip(signals, actual_prices[1:]):  # Shifted by one to align with buying/selling at the next price
        if signal == 1 and cash > 0:  # Buy
            stock_quantity = cash / price
            cash = 0
        elif signal == -1 and stock_quantity > 0:  # Sell
            cash = stock_quantity * price
            stock_quantity = 0

        portfolio_value = cash + (stock_quantity * price)
        portfolio_values.append(portfolio_value)
    
    return portfolio_values, buy_hold_value

# Function to execute the main process
def main():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=729)
    tickers = NEIGHBORS
    tickers.insert(0, TICKER_SYMBOL)
    dfs = []
    for ticker in tickers:
        df = fetch_data(ticker, start_date, end_date)
        dfs.append(df)

    df_merged = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Datetime'],
                                            how='outer'), dfs)
    
    X, y, scaler = prepare_data(df_merged, SEQUENCE_LENGTH, Y_INDEX)
    
    # Split data and reshape for CNN
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_cnn, X_test_cnn = reshape_for_cnn(X_train, X_test)

    # Build and train model
    model = cnn_bilstm_attention.CNNBiLSTMAttentionModel((SEQUENCE_LENGTH, 5, 1), (SEQUENCE_LENGTH, 5))
    train_model(model, X_train_cnn, X_train, y_train, X_test_cnn, X_test, y_test)
    
    # Generate and execute trading signals
    predictions = model.predict([X_test_cnn, X_test])
    predictions_inverse, y_test_inverse = inverse_transform(scaler, X_test, predictions, y_test)
    signals = generate_signals(predictions_inverse, y_test_inverse, THRESHOLD)
    portfolio_values, buy_hold_values = simulate_trading(signals, y_test_inverse, INITIAL_INVESTMENT)

    # Plot results
    plot_results(y_test_inverse, predictions_inverse, portfolio_values, buy_hold_values, signals)

    
# Define constants and configuration at the start
SEQUENCE_LENGTH = 60
INITIAL_INVESTMENT = 10000
TICKER_SYMBOL = "AAPL"
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
THRESHOLD = 0.005
Y_INDEX = 3
NEIGHBORS = []
# Call the main function to execute the process for the defined ticker
main()