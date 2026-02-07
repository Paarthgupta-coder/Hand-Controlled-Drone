import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Loading 4MB of data...")
# Read CSV and tell pandas to ignore bad lines and not look for a header
df = pd.read_csv('gesture_data.csv', header=None, on_bad_lines='skip')

# Drop rows that don't have enough columns (should be 43 columns for 21 landmarks + 1 label)
df = df.dropna()

# Separate labels (y) and coordinates (X)
y = df.iloc[:, 0].values.astype(str)
X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').values

# Final check: Remove any rows that became NaN during conversion
mask = ~np.any(np.isnan(X), axis=1)
X = X[mask].astype('float32')
y = y[mask]

print(f"Success! Final training set size: {len(X)} rows.")

# Process labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Neural Network Architecture (Meets the CNN/DNN requirement)
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nStarting Training...")
model.fit(X, y_encoded, epochs=30, batch_size=32, validation_split=0.2)

# Save
model.save('gesture_cnn_model.h5')
np.save('classes.npy', encoder.classes_)
print("\n[DONE] CNN model saved. You are ready for the demo!")