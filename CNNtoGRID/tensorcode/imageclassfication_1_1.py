import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras

data_dir = "./flower_photos/"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

class_names = train_ds.class_names

model = keras.models.load_model('imageclassfication.keras')
sunflower_path = "./Red_sunflower.jpg"
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(180, 180)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)