#  Recurrent neural network (RNN) Use the Google stock prices dataset and design a time series 
# analysis and prediction system using RNN.

# =========================
# 1. Import Libraries
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import yfinance as yf


# =========================
# 2. Load Dataset
# =========================
ticker = "GOOGL"
df = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Flatten columns (important for yfinance)
df.columns = df.columns.get_level_values(0)

# Use only 'Open' price
data = df[['Open']].values


# =========================
# 3. Train-Test Split
# =========================
train_size = int(len(data) * 0.8)

training_set = data[:train_size]
test_set = data[train_size:]


# =========================
# 4. Feature Scaling
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)


# =========================
# 5. Create Training Data (Windowing)
# =========================
X_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# =========================
# 6. Build RNN (LSTM Model)
# =========================
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))

model.add(Dense(1))  # Output layer

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')


# =========================
# 7. Train Model
# =========================
model.fit(X_train, y_train, epochs=20, batch_size=32)


# =========================
# 8. Test Data Preparation
# =========================
dataset_total = data

inputs = dataset_total[len(dataset_total) - len(test_set) - 60:]
inputs = inputs.reshape(-1, 1)

inputs = scaler.transform(inputs)

real_stock_price = test_set


# =========================
# 9. Prepare Test Sequences
# =========================
X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# =========================
# 10. Prediction
# =========================
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


# =========================
# 11. Visualization
# =========================
plt.plot(real_stock_price, color='red', label='Actual Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# ---------- REQUIREMENTS------------
# numpy
# pandas
# matplotlib
# scikit-learn
# tensorflow
# yfinance
