import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained VGG16 model and modify it for our classification task
def load_model():
    vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg16_model.layers:
        layer.trainable = False
    x = Flatten()(vgg16_model.output)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(132, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=vgg16_model.input, outputs=predictions)
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
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB mode
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
