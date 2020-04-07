import tensorflow as tf

from core.score_cam import ScoreCAM

IMAGE_PATH = "./cat.jpg"

if __name__ == "__main__":
    model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

    input_shape = (224, 224)

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=input_shape)
    img = tf.keras.preprocessing.image.img_to_array(img)

    model.summary()
    data = ([img], None)

    tabby_cat_class_index = 281
    explainer = ScoreCAM()
    # Compute ScoreCAM on VGG16
    image = explainer.explain(
        data, model, tabby_cat_class_index, input_shape, _grid=False
    )[0]
    explainer.save(image, ".", "grad_cam.png")