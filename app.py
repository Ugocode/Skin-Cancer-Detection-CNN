import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("melanoma_model.h5")

# Class names
class_names = ['Benign', 'Malignant']

# Prediction function
def predict(img):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {predicted_class: confidence}

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Ugocode Skin Cancer Detection (CNN) WEB APP",
    description="Upload a skin lesion image to detect if it's Benign or Malignant.",
    # examples=["example1.jpg", "example2.jpg"]  # Optional example images
)

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
