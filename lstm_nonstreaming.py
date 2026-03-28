import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- GPU CHECK ---
print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Success! Found GPU: {gpus[0].name}")
    # Optional: Memory growth prevents TF from grabbing all VRAM at once
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("CRITICAL ERROR: No GPU found. Terminating script to prevent CPU execution.")
    sys.exit(1) # Exit with error code
# -----------------

# 1. LOAD AND PREPARE DATA
filename = 'borg_traces_lstm_data.csv' 
df = pd.read_csv(filename)

feature_cols = ['cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 'priority', 'scheduling_class', 'rolling_cpu_load', 'rolling_failed_rate']
target_col = 'failed'

X_data = df[feature_cols].values
y_data = df[target_col].values

# 2. NORMALIZATION
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)

# 3. CREATE SLIDING WINDOW SEQUENCES
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

# 4. BUILD THE MODEL
# Note: CuDNN-optimized LSTMs require specific conditions (tanh activation, no recurrent dropout)
model = Sequential([
    # We add a tiny amount of recurrent_dropout (0.01)
    # This forces TensorFlow to avoid the CudnnRNN kernel that DirectML doesn't support.
    LSTM(units=128, 
         return_sequences=True, 
         input_shape=(WINDOW_SIZE, X.shape[2]),
         recurrent_dropout=0.01, # <--- THIS IS THE KEY FIX
         activation='tanh',
         recurrent_activation='sigmoid'),
    Dropout(0.2),
    
    LSTM(units=64, 
         recurrent_dropout=0.01, # <--- DO IT HERE TOO
         activation='tanh',
         recurrent_activation='sigmoid'),
    Dropout(0.2),
    
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. TRAIN THE MODEL
print(f"Training on {len(X_train)} samples using GPU...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=128, # Larger batches are usually faster on GPU
    validation_split=0.1, 
    verbose=1
)

# 6. EVALUATION & SAVE
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy:.2%}")
model.save('job_failure_predictor_gpu.h5')