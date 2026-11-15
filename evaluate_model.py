import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (128, 128)

# Load validation images
val_data = ImageDataGenerator(rescale=1./255)

val_generator = val_data.flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Load our trained model
model = tf.keras.models.load_model("model/retinal_model.keras")

# Evaluate
loss, accuracy = model.evaluate(val_generator)

print("\nValidation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Predictions
predictions = model.predict(val_generator)

predicted_classes = predictions.argmax(axis=1)
actual_classes = val_generator.classes

# Class names
class_names = list(val_generator.class_indices.keys())

print("\nClass Mapping:")
print(val_generator.class_indices)

print("\nClassification Report:")
print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=class_names
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual_classes,
        predicted_classes
    )
)