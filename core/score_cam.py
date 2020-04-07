"""
Core Module for Score-CAM Algorithm
"""

import tensorflow as tf
import numpy as np
import cv2

from utils.image import resize_activations, normalize_activations
from utils.display import grid_display, heatmap_display
from utils.saver import save_rgb

class ScoreCAM:

    """
    Perform Score-CAM algorithm for a given input

    Paper: [Score-CAM: Improved Visual Explanations Via Score-Wighted
            Class Activation Mapping](https://arxiv.org/abs/1910.01279v1)
    """

    def explain(
        self,
        validation_data,
        model,
        class_index,
        input_shape=None,
        colormap=cv2.COLORMAP_VIRIDIS,
        image_weight=0.7,
        _grid=True
    ):
        """
        Compute GradCAM for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            input_shape (Tuple[int, int]): Shape of input data, e.g. (224, 224)
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating the weight of
                the input image to be overlaying the calculated attribution maps. Defaults to `0.7`.
            _grid (bool): Whether display images on grid or separately.

        Returns:
            numpy.ndarray: Grid of all the GradCAM or 4D array (batch_size, height, width, channels)
        """
        assert input_shape != None, "Pass input shape argument"

        images, _ = validation_data
        assert images.shape[0] == 1, "Only one image per run supported"

        # according to section 4.1 of paper, we need the last convolutional layer
        layer_name = self.get_last_convolutional_layer_name(model)

        outputs, guided_grads = ScoreCAM.get_filters(
            model, images, layer_name, class_index, input_shape
        )

        cams = ScoreCAM.generate_ponderated_output(outputs, guided_grads)

        heatmaps = np.array(
            [
                # not showing the actual image if image_weight=0
                heatmap_display(cam.numpy(), image, colormap, image_weight)
                for cam, image in zip(cams, images)
            ]
        )

        if _grid:
            return grid_display(heatmaps)
        else:
            return heatmaps

    @staticmethod
    def get_last_convolutional_layer_name(model):
        """
        Search for the last convolutional layer to perform Score-CAM, as stated
        in section 4.1 in the original paper.

        Args:
            model (tf.keras.Model): tf.keras model to inspect

        Returns:
            str: Name of the target layer
        """
        for layer in reversed(model.layers):
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4 and layer.name.count('conv') > 0:
                return layer.name

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )
    
    @staticmethod
    def get_filters(model, images, layer_name, class_index, input_shape):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Last convolutional layer
            class_index (int): Index of targeted class
            input_shape (Tuple[int, int]): Shape of input data, e.g. (224, 224)

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """

        conv_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output]
        )
        softmax_model = tf.keras.models.Model(
            [model.inputs], [model.outputs]
        )

        inputs = tf.cast(images, tf.float32)

        conv_output = conv_model(inputs)
        resized_conv_output = resize_activations(conv_output[0], input_shape)
        normalized_maps = normalize_activations(resized_conv_output)

        # repeat input
        tiled_tensor = tf.tile(inputs, (normalized_maps.shape[0], 1, 1, 1))

        print(normalized_maps)
        print(tiled_tensor)

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """

        save_rgb(grid, output_dir, output_name)