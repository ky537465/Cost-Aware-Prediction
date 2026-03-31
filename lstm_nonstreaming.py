import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Success! Found GPU: {gpus[0].name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("CRITICAL ERROR: No GPU found. Terminating script to prevent CPU execution.")
    sys.exit(1)

filename = 'borg_traces_lstm_data.csv' 
df = pd.read_csv(filename)

feature_cols = ['cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 'priority', 'scheduling_class', 'rolling_cpu_load', 'rolling_failed_rate']
target_col = 'failed'

X_data = df[feature_cols].values
y_data = df[target_col].values

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)

def create_sequences(X_set, y_set, window_size=24):
    X_seq, y_seq = [], []
    for i in range(len(X_set) - window_size):
        X_seq.append(X_set[i : (i + window_size)])
        y_seq.append(y_set[i + window_size])
    return np.array(X_seq), np.array(y_seq)

WINDOW_SIZE = 24 
X, y = create_sequences(X_scaled, y_data, WINDOW_SIZE)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    LSTM(units=128, 
         return_sequences=True, 
         input_shape=(WINDOW_SIZE, X.shape[2]),
         recurrent_dropout=0.01,
         activation='tanh',
         recurrent_activation='sigmoid'),
    Dropout(0.2),
    
    LSTM(units=64, 
         recurrent_dropout=0.01,
         activation='tanh',
         recurrent_activation='sigmoid'),
    Dropout(0.2),
    
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(f"Training on {len(X_train)} samples using GPU...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=128,
    validation_split=0.1, 
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy:.2%}")
model.save('job_failure_predictor_gpu.h5')