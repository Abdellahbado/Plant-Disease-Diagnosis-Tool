import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

MODEL_PATH = "best_model.keras"
TRAIN_DATA_PATH = "/Users/mac/Documents/new AI/CV Projects/Plants Disease Classification/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"


def format_class_name(class_name):
    parts = class_name.split("___")

    plant = parts[0].capitalize()

    if len(parts) > 1:
        disease = parts[1].replace("_", " ").title()
        return f"{plant} - {disease}"
    else:
        return plant


def load_model_and_labels():
    model = load_model(MODEL_PATH)

    train_generator = ImageDataGenerator().flow_from_directory(
        TRAIN_DATA_PATH, target_size=(224, 224), batch_size=32, class_mode="categorical"
    )
    class_labels = list(train_generator.class_indices.keys())

    print(f"Number of classes: {len(class_labels)}")
    print(f"Class labels: {class_labels}")
    print(f"Model output shape: {model.output_shape}")

    return model, class_labels


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_disease(image, model, class_labels):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    print(f"Raw predictions shape: {predictions.shape}")
    print(f"Raw predictions: {predictions}")

    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predicted class index: {predicted_class}")

    if predicted_class < 0 or predicted_class >= len(class_labels):
        return f"Error: Predicted class {predicted_class} is out of range. Total classes: {len(class_labels)}"

    predicted_class_label = class_labels[predicted_class]
    formatted_class_label = format_class_name(predicted_class_label)
    return formatted_class_label
