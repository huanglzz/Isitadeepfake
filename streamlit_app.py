import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor
from huggingface_hub import login

# Login to Hugging Face
login(token="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")

# Load the model and processor
model_name = "thenewsupercell/deepfake-detection"  # Update this if necessary
processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Define the prediction function
def classify_image(image):
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Map the index to class labels (0: Fake, 1: Real)
    labels = {0: "Fake", 1: "Real"}
    predicted_label = labels[predicted_class_idx]

    return predicted_label

# Streamlit app
logo = "images/logo.svg"
st.image(logo, width=50)
st.title("IsItADeepfake - Deepfake Detection App")
  # Adjust width as needed

# Add logo
st.write("Upload an image to classify it as Real or Fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the image
    if st.button("Classify"):
        result = classify_image(image)
        
        # Improve the display of the result
        if result == "Real":
            st.success(f"🟢 **Prediction:** The image is **{result}**! This is a real image.")
        else:
            st.error(f"🔴 **Prediction:** The image is **{result}**! This is a fake image.")
