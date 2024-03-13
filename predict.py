import numpy as np
from tensorflow import keras

import read_data
from model import ResNet50
from model_old import CNN_MNIST


def main():
    # Load test data
    _, _, test_images, test_labels = read_data.fromGZ()

    # Load model
    # model = ResNet50((28, 28, 1), 10)
    model = CNN_MNIST()
    model.build(input_shape=(None, 28, 28, 1))
    model.load_weights("./model_weights.keras")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=["accuracy"],
    )
    model.summary()

    # Evaluate
    model.evaluate(test_images, test_labels, batch_size=10000)

    # Predict
    results = model.predict(test_images)
    results = np.argmax(results, axis=-1)

    # Output predict result
    with open("pred_results.csv", "w") as file:
        file.write("Id,Category\n")
        for i in range(len(results)):
            file.write(str(i) + "," + str(results[i]) + "\n")


if __name__ == "__main__":
    main()
