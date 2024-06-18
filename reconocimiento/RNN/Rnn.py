import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from preprocessing.PreProcesamiento import rellenar_caracteristicas

class RnnRecognizer(object):
    TRADUCTOR = {
        0: "uno",
        1: "dos",
        2: "tres",
        3: "cuatro",
    }

    def __init__(self, ruta_datos):
        self.X, self.y = self.__recuperar_datos(ruta_datos)
        self.__modelo = self.__crear_modelo()
        self.__max_tam_sec = len(self.X[0])

    def __crear_modelo(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        modelo = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=128),
            Dense(len(np.unique(self.y)), activation="softmax")
        ])

        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entrenar modelo
        modelo.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        return modelo

    def __recuperar_datos(self, ruta_datos):
        traductor_invertido = {v: k for (k, v) in self.TRADUCTOR.items()}
        datos = np.load(ruta_datos, allow_pickle=True)
        X = np.array(datos['features'], dtype=np.float32)
        y = datos['labels']
        y = np.array([traductor_invertido[dat] for dat in y], dtype=np.float32)
        return X, y

    def reconocer_palabra(self, mfccs):
        X = rellenar_caracteristicas([mfccs], self.__max_tam_sec, 20)
        X = np.array(X, dtype=np.float32)
        prediccion = self.__modelo.predict(X)
        indice = np.argmax(prediccion)
        return self.TRADUCTOR[indice]

