import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

import read_data
from randomErasing import randomEraserGenerator
from model import ResNet50, ResNet101
from model_old import CNN_MNIST


def main():
    # Data
    train_images, train_labels, test_images, test_labels = read_data.fromGZ()
    train_images_gen = randomEraserGenerator()

    # Training
    # model = ResNet50((28, 28, 1), 10)
    model = CNN_MNIST()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callback = keras.callbacks.ModelCheckpoint(
        filepath="model_weights.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )
    history = model.fit(
        # train_images_gen.run(train_images, train_labels, batch_size=1000),
        train_images,
        train_labels,
        batch_size=1500,
        epochs=3000,
        verbose=2,
        callbacks=[callback],
        validation_data=(test_images, test_labels),
        validation_batch_size=10000,
        # max_queue_size=10000,
        # workers=8,
        # use_multiprocessing=True,
    )

    # Plot history
    train_history = history.history["accuracy"]
    test_history = history.history["val_accuracy"]
    fig = plt.figure()
    (train_acc,) = plt.plot(train_history)
    (test_acc,) = plt.plot(test_history)
    plt.title(f"Accuracy")
    plt.xlabel("epoch")
    plt.legend(
        [train_acc, test_acc],
        [f"train_acc {train_history[-1]}", f"test_acc {test_history[-1]}"],
        loc="lower right",
    )
    plt.savefig(f"Accuracy.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
