from PIL import Image
import tensorflow as tf
import numpy as np

    
# Function to process image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Load image
    image = Image.open(image_path)

    image_np = np.asarray(image)
    image_p = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_p = tf.image.resize(image_p, (224, 224))
    image_p /= 255
    image_tensor = np.expand_dims(image_p, axis=0)

    return image_tensor