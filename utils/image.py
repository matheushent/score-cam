"""Module for image related operations"""
import tensorflow as tf

def resize_activations(enhanced_model_output, input_shape):
    """Utility function to resize a given tensor

    Args:
        enhanced_model_output:
        input_shape (tuple): shape of the input, e.g. (224, 224)

    Returns
        tensor (tf.Tensor): 4D-Tensor with shape (enhanced_model_output.shape[-1], H, W, K)
    """
    resized_activations = []
    for i in range(enhanced_model_output.shape[-1]):

        # resizing every activation map to original input image spatial dimensions
        resized_activations.append(tf.image.resize(enhanced_model_output[..., i], input_shape, preserve_aspect_ratio=True))

    return tf.stack(resized_activations)

def normalize_activations(tensor):
    """Utility function to normalize a given tensor

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, K)
    
    Returns
        tf.Tensor: 4D-Tensor with shape (batch_size, H, W, K) μ = 0 and σ = 1
    """
    
    normalized_tensor = tf.cast(
        tf.image.per_image_standardization(tensor), tf.uint8
    )

    return normalized_tensor