"""Module for image related operations"""
from skimage import transform
import tensorflow as tf
import numpy as np

def resize_activations(tensor, input_shape):
    """Utility function to resize a given tensor

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, K)
        input_shape (Tuple[int, int]): shape of the input, e.g. (224, 224)

    Returns
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, K, H, W)
    """

    resized_activations = list()

    for j in range(tensor.shape[0]):

        activations = list()

        for i in range(tensor.shape[-1]):
            activations.append(
                transform.resize(tensor[j, ..., i], input_shape, preserve_range=True)
            )
        
        resized_activations.append(np.array(activations))

    return tf.convert_to_tensor(np.array(resized_activations), dtype=tf.float32)

def normalize_activations(tensor):
    """Utility function to normalize a given tensor

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, K, H, W)
    
    Returns
        tf.Tensor: 4D-Tensor with shape (batch_size, K, H, W)
    """

    tensors = list()

    # goes through each image
    for i in range(tensor.shape[0]):
        flattened = tf.reshape(tensor[i], (tensor[i].shape[0], -1))

        max_a = tf.math.reduce_max(flattened, axis=1)
        min_a = tf.math.reduce_min(flattened, axis=1)

        diffs = tf.where(max_a > min_a, max_a - min_a, 1)

        normalized_tensor = (tensor[i] - tf.reshape(min_a, (-1, 1, 1))) / tf.reshape(diffs, (-1, 1, 1))

        tensors.append(normalized_tensor)
    
    return tf.stack(tensors, axis=0)