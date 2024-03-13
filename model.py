import tensorflow as tf
from tensorflow import keras


class ResidualBlock(keras.Model):
    def __init__(self, channel_in=64, channel_out=256):
        super().__init__()

        channel = channel_out // 4

        self.conv1 = keras.layers.Conv2D(channel, kernel_size=(1, 1), padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.av1 = keras.layers.Activation(tf.nn.relu)
        self.conv2 = keras.layers.Conv2D(channel, kernel_size=(3, 3), padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.av2 = keras.layers.Activation(tf.nn.relu)
        self.conv3 = keras.layers.Conv2D(
            channel_out, kernel_size=(1, 1), padding="same"
        )
        self.bn3 = keras.layers.BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = keras.layers.Add()
        self.av3 = keras.layers.Activation(tf.nn.relu)

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.av1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.av2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.av3(h)
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x: x
        else:
            return self._projection(channel_out)

    def _projection(self, channel_out):
        return keras.layers.Conv2D(channel_out, kernel_size=(1, 1), padding="same")


class ResNet50(keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self._layers = [
            # conv1
            keras.layers.Conv2D(
                64,
                input_shape=input_shape,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(tf.nn.relu),
            # conv2_x
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            ResidualBlock(64, 256),
            [ResidualBlock(256, 256) for _ in range(2)],
            # conv3_x
            keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(2, 2)),
            [ResidualBlock(512, 512) for _ in range(4)],
            # conv4_x
            keras.layers.Conv2D(1024, kernel_size=(1, 1), strides=(2, 2)),
            [ResidualBlock(1024, 1024) for _ in range(6)],
            # conv5_x
            keras.layers.Conv2D(2048, kernel_size=(1, 1), strides=(2, 2)),
            [ResidualBlock(2048, 2048) for _ in range(3)],
            # last part
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1000, activation=tf.nn.relu),
            keras.layers.Dense(output_dim, activation=tf.nn.softmax),
        ]

    def call(self, x):
        for layer in self._layers:
            if isinstance(layer, list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x

    def summary(self, input_shape=(28, 28, 1)):
        inputs = keras.Input(shape=input_shape)
        model = keras.Model(inputs=inputs, outputs=self.call(inputs))
        model.save("model_architecture.keras")
        model.summary()


class ResNet101(keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self._layers = [
            # conv1
            keras.layers.Conv2D(
                64,
                input_shape=input_shape,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding="same",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(tf.nn.relu),
            # conv2_x
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            ResidualBlock(64, 256),
            [ResidualBlock(256, 256) for _ in range(2)],
            # conv3_x
            keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(2, 2)),
            [ResidualBlock(512, 512) for _ in range(4)],
            # conv4_x
            keras.layers.Conv2D(1024, kernel_size=(1, 1), strides=(2, 2)),
            [ResidualBlock(1024, 1024) for _ in range(23)],
            # conv5_x
            keras.layers.Conv2D(2048, kernel_size=(1, 1), strides=(2, 2)),
            [ResidualBlock(2048, 2048) for _ in range(3)],
            # last part
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1000, activation=tf.nn.relu),
            keras.layers.Dense(output_dim, activation=tf.nn.softmax),
        ]

    def call(self, x):
        for layer in self._layers:
            if isinstance(layer, list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x

    def summary(self, input_shape=(28, 28, 1)):
        inputs = keras.Input(shape=input_shape)
        model = keras.Model(inputs=inputs, outputs=self.call(inputs))
        model.save("model_architecture.keras")
        model.summary()


if __name__ == "__main__":
    model = ResNet101((28, 28, 1), 10)
    model.summary()
