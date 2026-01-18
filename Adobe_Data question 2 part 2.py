import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Step 1: Loading the Adobe dataset from local CSV
raw_data = pd.read_csv("SectionB-Q2-Adobe_Data.csv", parse_dates=['Date'])
raw_data = raw_data.sort_values('Date').set_index('Date')
# Feature engineering: adding some juice to the data
raw_data['MA20_indicator'] = raw_data['Adj_Close'].rolling(window=20).mean()
raw_data['Price_Range'] = raw_data['High'] - raw_data['Low']
raw_data.dropna(inplace=True)
# Selection of features for the multivariate model
selected_cols = ['Adj_Close', 'Volume', 'MA20_indicator', 'Price_Range']
target_idx = 0 
# Normalizing everything between 0 and 1 for the LSTM
data_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_vals = data_scaler.fit_transform(raw_data[selected_cols])
# Function to chop data into sequences
def make_window_data(arr, window_size=20):
    train_x, train_y = [], []
    for i in range(window_size, len(arr)):
        train_x.append(arr[i-window_size:i, :])
        train_y.append(arr[i, target_idx])
    return np.array(train_x), np.array(train_y)
# Creating X and Y
X_full, y_full = make_window_data(scaled_vals, window_size=20)
# 85-15 split for training and testing
cutoff = int(len(X_full) * 0.85)
x_train_pts, x_test_pts = X_full[:cutoff], X_full[cutoff:]
y_train_pts, y_test_pts = y_full[:cutoff], y_full[cutoff:]
# --- Model 1: LSTM for temporal patterns ---
adobe_lstm_v2 = Sequential([
    Input(shape=(x_train_pts.shape[1], x_train_pts.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.25), # slightly adjusted dropout
    LSTM(32),
    Dense(1)
])
# --- Model 2: Basic ANN for baseline comparison ---
adobe_ann_v2 = Sequential([
    Input(shape=(x_train_pts.shape[1] * x_train_pts.shape[2],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
# Setup training parameters
adobe_lstm_v2.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
adobe_ann_v2.compile(optimizer='adam', loss='mse')
# Callback to prevent overfitting
early_stop_check = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
print("Starting the training process for both networks...")
# Running LSTM
adobe_lstm_v2.fit(x_train_pts, y_train_pts, validation_split=0.1, epochs=50, 
                  batch_size=32, callbacks=[early_stop_check], verbose=1)
# Running ANN
adobe_ann_v2.fit(x_train_pts.reshape(x_train_pts.shape[0], -1), y_train_pts, 
                 epochs=50, batch_size=32, verbose=0)
# Generating predictions
lstm_raw_preds = adobe_lstm_v2.predict(x_test_pts)
ann_raw_preds = adobe_ann_v2.predict(x_test_pts.reshape(x_test_pts.shape[0], -1))
# Helper to bring data back to dollar values
def reverse_scale(values_to_fix):
    temp_box = np.zeros((len(values_to_fix), len(selected_cols)))
    temp_box[:, target_idx] = values_to_fix.flatten()
    return data_scaler.inverse_transform(temp_box)[:, target_idx]
# Final prices
actual_prices = reverse_scale(y_test_pts)
lstm_final_prices = reverse_scale(lstm_raw_preds)
ann_final_prices = reverse_scale(ann_raw_preds)
# Calculate RMSE to see how we did
error_score = np.sqrt(mean_squared_error(actual_prices, lstm_final_prices))
print(f"\nFinal LSTM Model Error (RMSE): ${error_score:.2f}")
# Visualize the last 200 days
plt.figure(figsize=(14, 7))
plt.plot(actual_prices[-200:], label="Market Truth (Adobe)", color='black', alpha=0.9)
plt.plot(lstm_final_prices[-200:], label="LSTM Prediction", color='blue', ls='--')
plt.plot(ann_final_prices[-200:], label="ANN Prediction", color='darkred', alpha=0.5)
plt.title("Adobe Stock Price - Model Comparison Analysis")
plt.xlabel("Time Samples")
plt.ylabel("USD Price")
plt.legend()
plt.grid(alpha=0.2) # Adding a subtle grid
plt.show()
