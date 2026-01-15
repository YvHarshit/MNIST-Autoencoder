import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Load Data ------------------ #
def load_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_test  = x_test.astype("float32") / 255.

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    return x_train, x_test

# ------------------ Model ------------------ #
def build_autoencoder():
    input_img = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Flatten()(input_img)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    encoded = layers.Dense(32, activation="relu")(x)

    # Decoder
    x = layers.Dense(64, activation="relu")(encoded)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(28*28, activation="sigmoid")(x)
    decoded = layers.Reshape((28, 28, 1))(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder

# ------------------ Train ------------------ #
def train():
    x_train, x_test = load_data()
    model = build_autoencoder()

    model.fit(
        x_train, x_train,
        epochs=10,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    model.save("autoencoder.h5")
    return model, x_test

# ------------------ Visualize ------------------ #
def visualize(model, x_test):
    decoded_imgs = model.predict(x_test)

    n = 10
    plt.figure(figsize=(18, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.show()

# ------------------ Main ------------------ #
if __name__ == "__main__":
    model, x_test = train()
    visualize(model, x_test)
