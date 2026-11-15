import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Image size
IMG_SIZE = (128, 128)

# Load training images
train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical"
)

# Load validation images
val_data = ImageDataGenerator(rescale=1./255)

val_generator = val_data.flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical"
)

# Xception base model
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze Xception layers
base_model.trainable = False

# Add our own classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)

output = Dense(5, activation="softmax")(x)

# Create final model
model = Model(
    inputs=base_model.input,
    outputs=output
)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model structure
model.summary()

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save the trained model
model.save("model/retinal_model.keras")

print("Model training completed!")
print("Model saved successfully.")