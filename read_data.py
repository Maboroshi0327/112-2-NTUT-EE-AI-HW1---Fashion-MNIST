import pandas as pd
import numpy as np
from tensorflow import keras

import mnist_reader


def fromCSV():
    # Read the Fashion MNIST data from local CSV file
    df_train = pd.read_csv("./Fashion-MNIST/fashion-mnist_train.csv")
    df_test = pd.read_csv("./Fashion-MNIST/fashion-mnist_test.csv")

    # Get train images and labels from CSV file
    train_images = np.array(df_train.drop(["label"], axis=1), dtype="float32") / 255.0
    train_images = np.reshape(train_images, (60000, 28, 28, 1))
    train_labels = np.array(df_train["label"])
    train_labels = keras.utils.to_categorical(train_labels)

    test_images = np.array(df_test.drop(["label"], axis=1), dtype="float32") / 255.0
    test_images = np.reshape(test_images, (10000, 28, 28, 1))
    test_labels = np.array(df_test["label"])
    test_labels = keras.utils.to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def fromGZ():
    train_images, train_labels = mnist_reader.load_mnist(
        path="./Fashion-MNIST/",
        kind="train",
    )
    test_images, test_labels = mnist_reader.load_mnist(
        path="./Fashion-MNIST/",
        kind="t10k",
    )

    train_images = train_images.astype("float32") / 255.0
    train_images = np.reshape(train_images, (60000, 28, 28, 1))
    train_labels = keras.utils.to_categorical(train_labels)

    test_images = test_images.astype("float32") / 255.0
    test_images = np.reshape(test_images, (10000, 28, 28, 1))
    test_labels = keras.utils.to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def test():
    train_images, train_labels, test_images, test_labels = fromCSV()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

    train_images, train_labels, test_images, test_labels = fromGZ()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)


if __name__ == "__main__":
    test()
