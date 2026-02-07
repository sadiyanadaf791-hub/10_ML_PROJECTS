import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data_preprocessing import load_and_clean_data
from model import build_model

# Load data
data = load_and_clean_data("data/weather_data.csv")

X = data[['Latitude', 'Longitude', 'Magnitude']].values
y = (data['Magnitude'] > 4.5).astype(int)
y = to_categorical(y)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = build_model(X_train.shape[1])

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("✅ Test Accuracy:", acc)
