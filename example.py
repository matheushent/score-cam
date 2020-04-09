import tensorflow as tf
import numpy as np

from core.score_cam import ScoreCAM

IMAGE_PATH = "./cat.jpg"

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device('/GPU:0'):
    model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

    input_shape = (224, 224)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=input_shape)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    data = (img, None)

    tabby_cat_class_index = 281
    explainer = ScoreCAM()
    # Compute ScoreCAM on VGG16
    image = explainer.explain(
        data, model, tabby_cat_class_index, input_shape, _grid=False
    )[0]
    explainer.save(image, ".", "score_cam.png")