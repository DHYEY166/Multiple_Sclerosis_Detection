import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained VGG16 model and modify it for our classification task
def load_model():
    vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg16_model.layers:
        layer.trainable = False
    x = keras.layers.Flatten()(vgg16_model.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(.2)(x)
    x = keras.layers.Dense(132, activation='relu')(x)
    predictions = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.models.Model(inputs=vgg16_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Load the model
model = load_model()

# Streamlit Interface
st.title("Multiple Sclerosis Image Classification")
st.write("Upload an MRI image to classify it into one of the following categories:")
st.write("1. Control-Axial")
st.write("2. Control-Sagittal")
st.write("3. MS-Axial")
st.write("4. MS-Sagittal")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Make a prediction
    prediction = model.predict(image)
    prediction_class = np.argmax(prediction, axis=1)[0]

    # Map prediction to class label
    classes = ["Control-Axial", "Control-Sagittal", "MS-Axial", "MS-Sagittal"]
    predicted_class = classes[prediction_class]

    st.write(f"The model predicts that this image is: {predicted_class}")

    # Optionally, display the probability for each class
    st.write("Prediction probabilities:")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

# Display the accuracy and loss graphs (assuming you have the history object from the model training)
if 'history_16' in locals():
    history = history_16.history
    if 'accuracy' in history:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')
    
    if 'loss' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
    
    st.pyplot(plt)
