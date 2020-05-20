"""Module for image related operations"""
import tensorflow as tf

def resize_activations(enhanced_model_output, input_shape):
    """Utility function to resize a given tensor

    Args:
        enhanced_model_output (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, K)
        input_shape (Tuple[int, int]): shape of the input, e.g. (224, 224)

    Returns
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, K)
    """

    image = tf.image.resize(enhanced_model_output, input_shape, preserve_aspect_ratio=True)

    return image

def normalize_activations(tensor):
    """Utility function to normalize a given tensor

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, K)
    
    Returns
        tf.Tensor: 4D-Tensor with shape (batch_size, H, W, K) μ = 0 and σ = 1
    """

    normalized_tensor = tf.linalg.normalize(
        tensor, ord='euclidean', axis=None
    )[0]
    
    # ensure dtype == tf.float32
    if normalized_tensor.dtype != tf.float32:
        normalized_tensor = tf.cast(
            normalized_tensor, tf.float32
        )
    
    return normalized_tensor