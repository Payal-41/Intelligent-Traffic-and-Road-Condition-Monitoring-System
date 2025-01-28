import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model (update the path to your model file)
try:
    model = tf.keras.models.load_model('traffic_sign_classifier.h5')  # Replace with your model path
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define class names
class_names = ['Accident', 'dense_traffic', 'Fire','Sparse Traffic']  # Ensure these match your model's output
#class_names = ['Sparse Traffic', 'Fire', 'dense_traffic','Accident']  # Ensure these match your model's output

# Function to preprocess the image
def preprocess_image(image):
    try:
        img = image.resize((224, 224))  # Resize to match the model's input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        if img_array.ndim == 2:  # Grayscale image handling
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# App title and description
st.title("Traffic Scene Classification")
st.write("Upload an image of a traffic scene, and the app will predict its category (e.g., Sparse Traffic, Fire, Accident).")

# File uploader widget for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB format
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)
        if img_array is None:
            st.stop()

        # Predict the class using the trained model
        predictions = model.predict(img_array)
        st.write(f"Predictions (raw output): {predictions}")  # Debugging: Inspect raw model output

        if predictions.shape[1] != len(class_names):
            st.error("Mismatch between model output and class names. Check `class_names` or model.")
            st.stop()

        predicted_index = np.argmax(predictions[0])  # Get index of highest probability
        if predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
        else:
            st.error("Prediction index out of range. Check your model output or class_names.")
            st.stop()

        confidence = np.max(predictions[0])  # Get the confidence score for the predicted class

        # Display the predicted class and confidence score
        st.write(f"### Predicted Class: {predicted_class}")
        st.write(f"### Confidence: {confidence:.2f}")

        # Display confidence scores for all classes
        st.write("### Confidence Scores for All Classes:")
        for i, score in enumerate(predictions[0]):
            st.write(f"{class_names[i]}: {score:.2f}")

        # Display a bar chart of the confidence scores
        st.write("### Confidence Scores Visualization:")
        fig, ax = plt.subplots()
        ax.bar(class_names, predictions[0], color='skyblue')
        ax.set_ylabel("Confidence")
        ax.set_title("Prediction Confidence for Each Class")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
