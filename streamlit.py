import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title('🍅 Simple Tomato Leaf Disease Classifier')

# Load model (update path if needed)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('keras_potato_trained_model.h5')

model = load_model()

# Class names (update if your classes are different)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

uploaded_file = st.file_uploader('Upload a tomato leaf image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    st.success(f'Predicted Class: {class_names[pred_class]}') 