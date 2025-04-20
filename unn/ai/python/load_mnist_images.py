import tensorflow as tf
from numpy import reshape, uint8
import os

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Ask how many images to save
num_images_to_save = int(input("How many images to save? "))

# Create a folder to save the images
folder_name = "mnist\mnist_images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
else:
    print("Folder already exists. Continuing from previous download.")

# Get the current number of images in the folder
current_num_images = len(os.listdir(folder_name))

# Save the training images
for i in range(current_num_images, current_num_images + num_images_to_save):
    img = train_images[i]
    label = train_labels[i]
    filename = f"{folder_name}/train_{i}_{label}.png"
    img = img.astype(uint8)
    img = reshape(img, (28, 28, 1))  # Add a new axis for the channel
    img = tf.image.encode_png(img)
    tf.io.write_file(filename, img)

# Save the testing images
for i in range(current_num_images, current_num_images + num_images_to_save):
    img = test_images[i]
    label = test_labels[i]
    filename = f"{folder_name}/test_{i}_{label}.png"
    img = img.astype(uint8)
    img = reshape(img, (28, 28, 1))  # Add a new axis for the channel
    img = tf.image.encode_png(img)
    tf.io.write_file(filename, img)