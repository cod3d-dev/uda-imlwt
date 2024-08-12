import argparse

import img_utils as img_utils
import utilities as utils

import tensorflow as tf
import tensorflow_hub as hub

from keras import layers

# Create a parser for arguments needed to predict a class

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store', type=str, help='Path of flower image to predict')
parser.add_argument('model_path', action='store', type=str, default='model.keras', help='Path to model to use for inference')
parser.add_argument('--top_k', action='store', type=int, default=3, help='Return top K most likely categories')
parser.add_argument('--category_names', action='store', type=str, help='File that contains the categories of flowers')
# parser.add_argument('--gpu', action='store_true', default=False, help='use GPU or MPS (Apple Silicon)')
# parser.add_argument('--show_true_class', action='store_true', default=False, help='Show the true class of the image')


args = parser.parse_args()

image_path = args.image_path
model_path = args.model_path
top_k = args.top_k
category_names_path = args.category_names
# gpu = args.gpu


# Function to predict the class of the image using our model.
def predict(image_path, model_path, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load model from checkpoint
    model = tf.keras.models.load_model(model_path, custom_objects = {'KerasLayer':hub.KerasLayer})

    # Process image 
    img_tensor = img_utils.process_image(image_path) # Process the image

    prediction = model.predict(img_tensor)
     
    top_probs, top_classes = tf.nn.top_k(prediction, k=top_k)
    top_probs = list(top_probs.numpy()[0])
    top_classes = list(top_classes.numpy()[0])

    return top_probs, top_classes

top_probs, top_classes = predict(image_path, model_path, top_k)



# If the user specified a classes_names file, use it to load class names
if category_names_path:
    class_names = utils.load_classes(category_names_path)

    for i in range(len(top_classes)):
        print(f'Class: {top_classes[i]} - {class_names[str(top_classes[i])].title()}: {top_probs[i]*100:.2f}%')

else:
    for i in range(len(top_classes)):
        print(f'Class: {top_classes[i]}: {top_probs[i]*100:.2f}%')