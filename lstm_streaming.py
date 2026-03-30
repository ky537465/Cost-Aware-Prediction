import numpy as np
import os
from tensorflow.keras.callbacks import CSVLogger

# Global Variables
FILENAME = 'borg_traces_lstm_data.csv'
BATCH_SIZE = 128
WINDOW_SIZE = 24
FEATURE_COLS = ['cpu_req', 'mem_req', 'cpu_avg', 'mem_avg', 'priority', 'scheduling_class', 'rolling_cpu_load', 'rolling_failed_rate']
TARGET_COL = 'failed'
SELECT_COLS = FEATURE_COLS + [TARGET_COL]

# Streaming reads csv in chunks
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=1,
        select_columns=SELECT_COLS,
        label_name=TARGET_COL,
        num_epochs=1,
        ignore_errors=True
    )

    # Convert dictionary to a tensor
    def pack_features(features, label):
        features_tensor = tf.stack([tf.cast(features[col], tf.float32) for col in FEATURE_COLS], axis=-1)
        return tf.squeeze(features_tensor, axis=0), label

    dataset = dataset.map(pack_features)
    dataset = dataset.window(WINDOW_SIZE, shift=1, stride=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(WINDOW_SIZE), y.skip(WINDOW_SIZE-1).take(1))))
    return dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Setting up the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, len(FEATURE_COLS)),
                         recurrent_dropout=0.01, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, recurrent_dropout=0.01, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Logging data
csv_logger = CSVLogger('training_results.csv', append=False, separator=',')

# Training
streamed_train_ds = get_dataset(FILENAME)

print("Starting streaming training and logging results...")
history = model.fit(
    streamed_train_ds,
    epochs=20,
    callbacks=[csv_logger]
)

# Final output
final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]

print("-" * 30)
print(f"Training Complete!")
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Accuracy: {final_accuracy:.2%}")
print("Full epoch-by-epoch results saved to: training_results.csv")

model.save('job_failure_predictor_streaming.h5') 