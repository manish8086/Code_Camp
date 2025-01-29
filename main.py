import os
import tensorflow as tf
import tensorflow_cloud as tfc

# Entry point for scaling TensorFlow models
if __name__ == "__main__":
    print("Starting TensorFlow Cloud Project...")

    # Ensure the 'scripts' directory exists
    os.makedirs("scripts", exist_ok=True)

    # Example TensorFlow Keras model training script (mnist_example.py)
    with open("mnist_example.py", "w") as f:
        f.write("""\
import tensorflow as tf

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28)).astype('float32') / 255

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)
""")

    # Create and write the scaling script (scale_mnist.py)
    with open("scripts/scale_mnist.py", "w") as f:
        f.write("""\
import tensorflow_cloud as tfc

def scale_model():
    tfc.run(entry_point="mnist_example.py")

if __name__ == "__main__":
    scale_model()
""")

    # Notify that the scaling script has been created
    print("Scaling script created successfully.")
