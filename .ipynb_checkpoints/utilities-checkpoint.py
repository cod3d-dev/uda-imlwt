import json

# Function to load the classes names from json file
def load_classes(file_path):
    with open(file_path, 'r') as f:
        class_names = json.load(f)
        return class_names