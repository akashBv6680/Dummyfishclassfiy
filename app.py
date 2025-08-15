import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests

# --------------------
# 1. App Configuration
# --------------------
st.set_page_config(page_title="Species Predictor", layout="centered")

st.title("Image-Based Species Predictor")
st.markdown("Upload an image, and I'll predict the species using a powerful AI model.")
st.write("---")

# --------------------
# 2. Model Loading
# --------------------
# This function loads the model and processor from Hugging Face.
# We use st.cache_data to make sure the model is only downloaded and
# loaded once, which makes the app run much faster.
@st.cache_data
def load_model():
    """Loads a pre-trained image classification model and its processor."""
    # We'll use a pre-trained ResNet model, a standard choice for image tasks.
    # The model is trained on a huge dataset, so it can classify many objects.
    model_name = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()
id_to_label = model.config.id2label
# A loading message to inform the user the model is being prepared.
st.success("Model loaded and ready for prediction!")

# --------------------
# 3. User Interface for Image Upload
# --------------------
# This creates a file uploader widget in the Streamlit app.
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a plant, animal, or object to get a prediction."
)

# --------------------
# 4. Prediction Logic
# --------------------
if uploaded_file is not None:
    # If a file is uploaded, we process it.
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Predicting...")

    try:
        # Open the image using Pillow.
        image = Image.open(uploaded_file).convert("RGB")

        # The processor prepares the image for the model (resizing, normalization).
        inputs = processor(images=image, return_tensors="pt")

        # Make the prediction. We don't need to compute gradients here.
        with torch.no_grad():
            outputs = model(**inputs)

        # The model's output is a set of "logits". We find the one with the
        # highest value to get the predicted class index.
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        # Look up the human-readable label for the predicted index.
        predicted_species = id_to_label[predicted_class_idx]
        
        # Display the result to the user.
        st.write(f"I predict this is a: **{predicted_species}**")
        st.balloons() # Add a celebratory animation!

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

