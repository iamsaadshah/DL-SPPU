# 🚀 Deep Learning Assignment – IMDB Sentiment Classification (DNN)

# ================================
# 1. IMPORT LIBRARIES
# ================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ================================
# 2. LOAD DATASET (IMDB)
# ================================
# Only keep top 10,000 most frequent words
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ================================
# 3. DATA PREPROCESSING
# ================================
# Convert sequences into fixed-size binary vectors (multi-hot encoding)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
        
    return results

X_train = vectorize_sequences(X_train)
X_test = vectorize_sequences(X_test)

# Convert labels to float
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

# ================================
# 4. BUILD DEEP NEURAL NETWORK
# ================================
model = Sequential()

# Input Layer + Hidden Layer 1
model.add(Dense(32, activation='relu', input_shape=(10000,)))
model.add(Dropout(0.2))

# Hidden Layer 2
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))

# Output Layer (Binary Classification)
model.add(Dense(1, activation='sigmoid'))

# ================================
# 5. COMPILE MODEL
# ================================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================================
# 6. TRAIN MODEL
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=512,
    validation_split=0.2
)

# ================================
# 7. EVALUATE MODEL
# ================================
loss, accuracy = model.evaluate(X_test, y_test)

print("\nFinal Test Loss:", loss)
print("Final Test Accuracy:", accuracy)

# ================================
# 8. MAKE PREDICTIONS
# ================================
predictions = model.predict(X_test)

# Convert probabilities to binary labels
predicted_labels = (predictions > 0.5).astype(int)

print("\nSample Predictions:")
print("Predicted:", predicted_labels[:10].flatten())
print("Actual   :", y_test[:10])

# ================================
# 9. VISUALIZATION
# ================================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()