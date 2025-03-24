import tensorflow as tf
from tensorflow.keras import layers

class IntentModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Embedding(input_dim=10000, output_dim=64),
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def get_model(self):
        return self.model