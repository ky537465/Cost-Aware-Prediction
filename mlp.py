import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

gray_scale = 255

x_train = x_train.astype('float32') / gray_scale
x_test = x_test.astype('float32') / gray_scale

print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)

fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28), aspect='auto')
        k += 1
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='sigmoid'),  
    tf.keras.layers.Dense(128, activation='sigmoid'), 
    tf.keras.layers.Dense(10, activation='softmax'),  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_acc = []
train_loss = []
batch_size = 32

print("Starting streaming updates...")

for i in range(0, len(x_train), batch_size):
    x_stream = x_train[i:i+batch_size]
    y_stream = y_train[i:i+batch_size]
    
    metrics = model.train_on_batch(x_stream, y_stream)
    
    if i % (batch_size * 100) == 0:
        train_loss.append(metrics[0])
        train_acc.append(metrics[1])
        print(f"Sample {i}/{len(x_train)} - Loss: {metrics[0]:.4f}, Acc: {metrics[1]:.4f}")

print("Streaming complete.")

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Streaming Accuracy', color='blue')
plt.title('Online Learning Accuracy', fontsize=14)
plt.xlabel('Logged Batches', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Streaming Loss', color='red')
plt.title('Online Learning Loss', fontsize=14)
plt.xlabel('Logged Batches', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()