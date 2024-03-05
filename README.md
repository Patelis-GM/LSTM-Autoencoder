
# Stock Anomaly Detection using LSTM Encoder-Decoder Architecture
This project was developed for the "Algorithms in Software Development" course at the Department of Informatics and Telecommunications of the National and Kapodistrian University of Athens. It focuses on detecting anomalies in stock prices utilizing a Long Short-Term Memory (LSTM) recurrent neural network with an encoder-decoder architecture. The model is implemented using TensorFlow framework and utilizes historical stock price data from Yahoo Finance.

# Project Overview
The project consists of two main Python files:

detectTrain.py: This script is responsible for the training process of the LSTM model. It requires a mandatory "-d" argument which should be the path of a CSV file containing historical stock prices. Each entry in this file should be in the format: STOCK_NAME stock_price_day_1, stock_price_day_2, ..., stock_price_day_n. Note that the model is trained on 80% of each stock's price data.

detect.py: This script demonstrates the usage of the trained model for anomaly detection. It requires the following arguments:

- Mandatory "-d": Path of the CSV file containing historical stock prices, with each entry in the format mentioned above.
- Mandatory "-n": Number of stocks for which anomaly detection will be performed.
- Mandatory "-mae": Threshold value (float) considered as an anomaly.
- Optional "-r": Boolean argument (true or false) indicating whether to reconstruct the corresponding curve for anomaly detection.

# Requirements
Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Pandas
- NumPy
