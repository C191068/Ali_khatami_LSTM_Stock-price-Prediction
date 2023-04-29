## Ali_khatami_LSTM_Stock-price-Prediction
###Code

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


df=pd.read_csv("/content/drive/MyDrive/AAPL.csv")

df

# Convert the date column to a datetime object
df["Date"] = pd.to_datetime(df["Date"])

# Sort the data by date
df = df.sort_values("Date")

df


# Set the date column as the index of the dataframe

df.set_index("Date", inplace=True)

df



# Extract the closing price column
data = df[["Close"]].values

data




# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
np.set_printoptions(suppress=True, precision=10)
data


#The purpose of this function is to create a dataset suitable for training a time series prediction model, such as an LSTM neural network
def create_time_series(data, lookback=60, predict_steps=1):
    X, y = [], []
    for i in range(lookback, len(data) - predict_steps + 1):
        X.append(data[i - lookback:i])
        y.append(data[i:i + predict_steps])
    return np.array(X), np.array(y)
    
    
    
#this code defines a complex LSTM neural network model for time series prediction, consisting of four LSTM layers with varying numbers of units and Dropout layers to reduce overfitting. 
model = Sequential()
model.add(LSTM(50, input_shape=(60, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")


#this code prepares the training data for an LSTM neural network by selecting the first 2000 samples of the data array and using the create_time_series() function to generate a set of input and target time series data
train_data = data[:2000]

X_train, y_train = create_time_series(train_data, lookback=60)


#this code trains the LSTM neural network model using the training data X_train and y_train for 50 epochs with a batch size of 64.
model.fit(X_train, y_train, epochs=60, batch_size=64)


# Create the time-series data for testing
test_data = data[2000:]
X_test, y_test = create_time_series(test_data, lookback=60)


# Make predictions on the test data
y_pred = model.predict(X_test)


# Inverse transform the scaled data
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]


# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)



# Print the mean absolute error
print("MAE:", mae)



# Plot the actual and predicted prices
plt.plot(y_test, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()



Here in the code above the Mean absolute error is 8.4187753210999. please do something so that the mean absolute error is minimized to about zero

```




