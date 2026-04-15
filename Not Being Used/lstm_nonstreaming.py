import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import Callback

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("ERROR: No GPU detected. System exiting as requested.")
    sys.exit(1)
else:
    print(f"GPU detected: {gpus}")

FILENAME = 'borg_traces_lstm_data.csv'
RESULTS_FILE = 'lstm_nonstreaming_results.csv'
BATCH_SIZE = 128
WINDOW_SIZE = 24
FEATURE_COLS = ['cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 'priority', 'scheduling_class', 'rolling_cpu_load', 'rolling_failed_rate']
TARGET_COL = 'failed'
SELECT_COLS = FEATURE_COLS + [TARGET_COL]

df = pd.read_csv(FILENAME)
X_data = df[FEATURE_COLS].values
y_data = df[TARGET_COL].values

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)

def create_sequences(X_set, y_set, window_size=24):
    X_seq, y_seq = [], []
    for i in range(len(X_set) - window_size):
        X_seq.append(X_set[i : (i + window_size)])
        y_seq.append(y_set[i + window_size])
    return np.array(X_seq), np.array(y_seq)

X, y = create_sequences(X_scaled, y_data, WINDOW_SIZE)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_val_sample = X_test[:BATCH_SIZE]
y_val_sample = y_test[:BATCH_SIZE]

class DetailedResultsLogger(Callback):
    def __init__(self, filename, val_x, val_y):
        super().__init__()
        self.filename = filename
        self.val_x = val_x
        self.val_y = val_y
        with open(self.filename, 'w') as f:
            f.write("epoch,loss,accuracy,val_loss,val_accuracy,actual_failed,predicted_failed\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get predictions for the sample validation set
        preds = self.model.predict(self.val_x, verbose=0)
        actuals = self.val_y.flatten()
        predictions = (preds.flatten() > 0.5).astype(int)
        
        with open(self.filename, 'a') as f:
            for a, p in zip(actuals, predictions):
                f.write(f"{epoch},{logs.get('loss', 0):.4f},{logs.get('accuracy', 0):.4f},"
                        f"{logs.get('val_loss', 0):.4f},{logs.get('val_accuracy', 0):.4f},"
                        f"{a},{p}\n")

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

detailed_logger = DetailedResultsLogger(RESULTS_FILE, X_val_sample, y_val_sample)

print(f"Training on {len(X_train)} samples using GPU...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=BATCH_SIZE,
    validation_split=0.1, 
    verbose=1,
    callbacks=[detailed_logger]
)

loss, accuracy = model.evaluate(X_test, y_test)
model.save('lstm_job_predictor_nonstreaming.h5')
print(f"Results saved to {RESULTS_FILE}")