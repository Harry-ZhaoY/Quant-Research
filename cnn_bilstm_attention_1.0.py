import numpy as np
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Bidirectional, LSTM, Attention, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping

# Function to fetch market data
def fetch_data(ticker_symbol, start_date, end_date):
    ticker_data = yf.Ticker(ticker_symbol)
    return ticker_data.history(interval='1h', start=start_date, end=end_date)[['Open', 'High', 'Low', 'Close', 'Volume']]

# Function to normalize data and create sequences
def prepare_data(df, sequence_length=70):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df)
    
    xs, ys = [], []
    for i in range(len(scaled_features) - sequence_length):
        x = scaled_features[i:(i + sequence_length)]
        y = scaled_features[i + sequence_length, 3]  # Assuming 'Close' is at index 3
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

# Build CNN-BiLSTM model
def build_cnn_bilstm_model(input_shape_cnn, input_shape_lstm):
     # CNN Layer
    input_cnn = Input(shape=input_shape_cnn)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_cnn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    flatten_cnn = Flatten()(pool1)
    dense_cnn = Dense(64, activation='relu')(flatten_cnn)

    # LSTM Layer
    input_lstm = Input(shape=input_shape_lstm)
    lstm_out = LSTM(64, return_sequences=True)(input_lstm)

    # Instead of flattening LSTM output, we should directly use it with attention
    # This ensures we have the [batch_size, sequence_length, features] shape

    # Attention Mechanism - Adjusted for proper input shapes
    # Assuming sequence_length for lstm_out is preserved to be compatible with Attention
    query_value_inputs = lstm_out  # Directly using LSTM output

    # Apply attention
    attention_out = Attention()([query_value_inputs, query_value_inputs])
    
    # After attention, you can decide to flatten or further process depending on the next steps
    attention_flatten = Flatten()(attention_out)

    # Combine the CNN and attention processed LSTM outputs
    combined_out = Concatenate(axis=-1)([dense_cnn, attention_flatten])
    
    # Final output layer
    dense_final = Dense(64, activation='relu')(combined_out)
    final_output = Dense(1)(dense_final)

    model = Model(inputs=[input_cnn, input_lstm], outputs=final_output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(
        [X_train_cnn, X_train],  # Inputs for the CNN and LSTM
        y_train,  # Training targets
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X_test_cnn, X_test], y_test),  # Validation data
        callbacks=[early_stopping]
    )

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
    for predicted, actual in zip(predictions[1:], actual_prices[:-1]):  # Exclude the last actual price since there's no prediction for the next day
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
def main(ticker_symbol):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=729)
    df = fetch_data(ticker_symbol, start_date, end_date)
    
    X, y, scaler = prepare_data(df, SEQUENCE_LENGTH)
    
    # Split data and reshape for CNN
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_cnn, X_test_cnn = reshape_for_cnn(X_train, X_test)

    # Build and train model
    model = build_cnn_bilstm_model((SEQUENCE_LENGTH, 5, 1), (SEQUENCE_LENGTH, 5))
    train_model(model, X_train_cnn, X_train, y_train, X_test_cnn, X_test, y_test)
    
    # Generate and execute trading signals
    predictions = model.predict([X_test_cnn, X_test])
    predictions_inverse, y_test_inverse = inverse_transform(scaler, X_test, predictions, y_test)
    signals = generate_signals(predictions_inverse, y_test_inverse)
    portfolio_values, buy_hold_values = simulate_trading(signals, y_test_inverse, INITIAL_INVESTMENT)

    # Plot results
    plot_results(y_test_inverse, predictions_inverse, portfolio_values, buy_hold_values, signals)

    
# Define constants and configuration at the start
SEQUENCE_LENGTH = 15
INITIAL_INVESTMENT = 10000
TICKER_SYMBOL = "SPY"
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
THRESHOLD = 0.005
# Call the main function to execute the process for the defined ticker
main(TICKER_SYMBOL)