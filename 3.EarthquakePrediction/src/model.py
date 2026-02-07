from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
