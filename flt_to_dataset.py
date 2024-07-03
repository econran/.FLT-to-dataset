import os
import numpy as np
import tensorflow as tf
import pdb

image_dir = "/data/CT_images/train/images"

image_height = 512  # Adjust as needed
image_width = 512  # Adjust as needed

# Define image dimensions (assuming all images are of the same size)
image_height = 512  # Adjust as needed
image_width = 512  # Adjust as needed

# Load images into a list using numpy.fromfile
image_list = []
for filename in os.listdir(image_dir):
    if filename.endswith('.flt'):
        file_path = os.path.join(image_dir, filename)
        # Read the binary file into a numpy array
        image = np.fromfile(file_path, dtype=np.float32)  # Adjust dtype if necessary
        # Reshape the array to the correct dimensions
        image = image.reshape((image_height, image_width))
        image_list.append(image)

pdb.set_trace()
# Convert the list to a numpy array
images_np = np.array(image_list)

# Create a TensorFlow dataset from the numpy array
flt_to_imgs_dataset = tf.data.Dataset.from_tensor_slices(images_np)

# Print the dataset elements
#for image in dataset:
   # print(image)
