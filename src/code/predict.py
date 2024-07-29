import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

def load_best_model():
    model_path = "models/selected_model"
    return keras.models.load_model(model_path)

def prepare_data():
    datasets, _ = tfds.load(name='fashion_mnist', with_info=True, as_supervised=True)
    return datasets['test'].map(scale).batch(1)

def predict(model, data):
    for image, label in data.take(1):  # Predict on a single image
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        actual_class = label.numpy()[0]
        print(f"Predicted class: {predicted_class}")
        print(f"Actual class: {actual_class}")

if __name__ == "__main__":
    model = load_best_model()
    data = prepare_data()
    predict(model, data)