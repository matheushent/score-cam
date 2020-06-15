"""
Callback module for Score-CAM
"""
from datetime import datetime
from pathlib import Path

from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np

from core.score_cam import ScoreCAM

class ScoreCAMCallback(Callback):
    """
    Perform Score-CAM algorithm for a given input

    Paper: [Score-CAM:Improved Visual Explanations Via
            Score-Weighted Class Activation Mapping](https://arxiv.org/abs/1910.01279v1)
    """

    def __init__(
        self,
        validation_data,
        class_index,
        input_shape,
        output_dir=Path(".logs/score_cam")
    ):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                    to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            input_shape (Tuple): Shape of input tensor, e.g. (224, 224) for VGG16
            layer_name (str): Targeted layer for Score-CAM
            output_dir (str): Output directory path
        """
        super(ScoreCAMCallback, self).__init__()
        self.validation_data = validation_data
        self.class_index = class_index
        self.input_shape = input_shape
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw Score-CAM outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        explainer = ScoreCAM()
        heatmap = explainer.explain(
            self.validation_data,
            self.model,
            self.class_index,
            self.input_shape
        )

        # Using the file writer, log the reshaped image
        with self.file_writer.as_default():
            tf.summary.image("Score-CAM", np.array([heatmap]), step=epoch)