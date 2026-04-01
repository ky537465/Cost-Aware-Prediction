import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_DISABLE_CUDNN_RNN'] = '1'

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("ERROR: No GPU detected. System exiting as requested.")
    sys.exit(1)
else:
    print(f"GPU detected: {gpus}")

FILENAME = 'borg_traces_lstm_data.csv'
RESULTS_FILE = 'lstm_streaming_results.csv'
BATCH_SIZE = 128
WINDOW_SIZE = 24
FEATURE_COLS = ['cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 'priority', 'scheduling_class', 'rolling_cpu_load', 'rolling_failed_rate']
TARGET_COL = 'failed'
SELECT_COLS = FEATURE_COLS + [TARGET_COL]

def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=1,
        select_columns=SELECT_COLS,
        label_name=TARGET_COL,
        num_epochs=1,
        ignore_errors=True
    )

    def pack_features(features, label):
        features_tensor = tf.stack([tf.cast(features[col], tf.float32) for col in FEATURE_COLS], axis=-1)
        # remove the batch dimension from make_csv_dataset
        return tf.squeeze(features_tensor, axis=0), label

    dataset = dataset.map(pack_features)
    dataset = dataset.window(WINDOW_SIZE, shift=1, stride=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((
        x.batch(WINDOW_SIZE), 
        y.skip(WINDOW_SIZE - 1).take(1)
    )))
    
    return dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class DetailedResultsLogger(Callback):
    def __init__(self, filename, validation_data):
        super().__init__()
        self.filename = filename
        self.validation_data = validation_data
        with open(self.filename, 'w') as f:
            f.write("epoch,loss,accuracy,val_loss,val_accuracy,actual_failed,predicted_failed\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for inputs, labels in self.validation_data.take(1):
            preds = self.model.predict(inputs, verbose=0)
            actuals = labels.numpy().flatten()
            predictions = (preds.flatten() > 0.5).astype(int)
            
            with open(self.filename, 'a') as f:
                for a, p in zip(actuals, predictions):
                    f.write(f"{epoch},{logs.get('loss', 0):.4f},{logs.get('accuracy', 0):.4f},"
                            f"{logs.get('val_loss', 0):.4f},{logs.get('val_accuracy', 0):.4f},"
                            f"{a},{p}\n")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, len(FEATURE_COLS)),
                         recurrent_dropout=0.001, activation='tanh'), 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, activation='tanh', recurrent_dropout=0.001),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

full_ds = get_dataset(FILENAME)
val_ds = full_ds.take(10)
train_ds = full_ds.skip(10)

detailed_logger = DetailedResultsLogger(RESULTS_FILE, val_ds)

print("Starting streaming training...")
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[detailed_logger]
)

model.save('job_failure_predictor_streaming.h5')
print(f"Results saved to {RESULTS_FILE}")