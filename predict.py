import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("model/retinal_model.keras")

# Class names
class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]

# Change this to the path of one retinal image
import os

image_folder = "dataset/val/Mild"

image_name = os.listdir(image_folder)[0]

image_path = os.path.join(image_folder, image_name)

print("Testing image:", image_path)

# Load image
img = image.load_img(
    image_path,
    target_size=(128, 128)
)

# Convert image to array
img_array = image.img_to_array(img)

# Normalize pixels
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)

# Get predicted class
predicted_index = np.argmax(prediction[0])

predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("Predicted Class:", predicted_class)
print("Confidence:", round(confidence, 2), "%")