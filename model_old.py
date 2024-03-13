from tensorflow import keras


class CNN_MNIST(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1_1 = keras.layers.Conv2D(
            filters=6,
            kernel_size=7,
            padding="same",
            activation="relu",
        )
        self.drop1_1 = keras.layers.Dropout(rate=0.4)
        self.bn1_1 = keras.layers.BatchNormalization()

        self.conv1_2 = keras.layers.Conv2D(
            filters=3,
            kernel_size=7,
            padding="same",
            activation="relu",
        )
        self.drop1_2 = keras.layers.Dropout(rate=0.3)
        self.bn1_2 = keras.layers.BatchNormalization()

        self.conv2_1 = keras.layers.Conv2D(
            filters=6,
            kernel_size=5,
            padding="same",
            activation="relu",
        )
        self.drop2_1 = keras.layers.Dropout(rate=0.4)
        self.bn2_1 = keras.layers.BatchNormalization()

        self.conv2_2 = keras.layers.Conv2D(
            filters=3,
            kernel_size=5,
            padding="same",
            activation="relu",
        )
        self.drop2_2 = keras.layers.Dropout(rate=0.3)
        self.bn2_2 = keras.layers.BatchNormalization()

        self.conv3_1 = keras.layers.Conv2D(
            filters=6,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
        self.drop3_1 = keras.layers.Dropout(rate=0.4)
        self.bn3_1 = keras.layers.BatchNormalization()

        self.conv3_2 = keras.layers.Conv2D(
            filters=3,
            kernel_size=3,
            padding="same",
            activation="relu",
        )
        self.drop3_2 = keras.layers.Dropout(rate=0.3)
        self.bn3_2 = keras.layers.BatchNormalization()

        self.pool = keras.layers.AveragePooling2D(pool_size=(2, 2))

        self.flatten = keras.layers.Flatten()
        self.relu1 = keras.layers.ReLU()
        self.drop4 = keras.layers.Dropout(rate=0.35)

        self.fc1 = keras.layers.Dense(
            units=300,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2=0.0005),
        )
        self.drop5 = keras.layers.Dropout(rate=0.3)

        self.out = keras.layers.Dense(
            units=10,
            activation="softmax",
            kernel_regularizer=keras.regularizers.l2(l2=0.0005),
        )

    def call(self, x):
        x1 = self.conv1_1(x)
        x1 = self.drop1_1(x1)
        x1 = self.bn1_1(x1)

        x2 = self.conv2_1(x)
        x2 = self.drop2_1(x2)
        x2 = self.bn2_1(x2)

        x3 = self.conv3_1(x)
        x3 = self.drop3_1(x3)
        x3 = self.bn3_1(x3)

        x4 = keras.layers.Concatenate(axis=3)([x1, x2, x3])

        x1 = self.conv1_2(x4)
        x1 = self.drop1_2(x1)
        x1 = self.bn1_2(x1)

        x2 = self.conv2_2(x4)
        x2 = self.drop2_2(x2)
        x2 = self.bn2_2(x2)

        x3 = self.conv3_2(x4)
        x3 = self.drop3_2(x3)
        x3 = self.bn3_2(x3)

        x4 = keras.layers.Concatenate(axis=3)([x1, x2, x3])
        x4 = self.pool(x4)

        x4 = self.flatten(x4)
        x4 = self.relu1(x4)
        x4 = self.drop4(x4)

        x4 = self.fc1(x4)
        x4 = self.drop5(x4)
        x4 = self.out(x4)

        return x4

    def summary(self, input_shape=(28, 28, 1)):
        inputs = keras.Input(shape=input_shape)
        model = keras.Model(inputs=inputs, outputs=self.call(inputs))
        model.save("model_architecture.keras")
        model.summary()


if __name__ == "__main__":
    model = CNN_MNIST()
    model.summary()
