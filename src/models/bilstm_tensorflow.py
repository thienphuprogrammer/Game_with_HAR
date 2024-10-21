from keras.layers import Normalization
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional


class BiLSTMTensorflow:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()
        model.add(
            Bidirectional(
                LSTM(1024, return_sequences=True),
                input_shape=(self.input_shape[1], self.input_shape[2]),
            )
        )
        model.add(Dropout(0.5))  # Dropout rate reduced
        # Normalize the data
        model.add(Normalization())
        model.add(Bidirectional(LSTM(512, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Normalization())
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Normalization())
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.5))
        model.add(Normalization())
        model.add(Dense(6, activation="softmax"))
        return model

    def compile_model(self):
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        return self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
