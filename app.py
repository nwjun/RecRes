import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

Classes = ['Zucchini', 'Yoghurt', 'Tomato', 'Soyghurt', 'Sour-Cream', 'Satsumas', 'Red-Grapfruit', 'Red-Beet',
           'Potato', 'Pomegranate', 'Pineapple', 'Pepper', 'Pear', 'Peach', 'Passion-Fruit', 'Orange', 'Onion',
           'Oatghurt', 'Oat-Milk', 'Milk', 'Melon', 'Mango', 'Lime', 'Lemon', 'Leek', 'Kiwi', 'Juice', 'Ginger',
           'Cucumber', 'Carrots', 'Cabbage', 'Brown-Cap-Mushroom', 'Banana', 'Avocado', 'Aubergine', 'Asparagus',
           'Apple']

# Reverse the Classes list
Classes_reversed = Classes[::-1]


# Load the model
@st.cache_resource
def load_model():
    model_path = "/Users/nwjun/Downloads/resnet50_frozen_model"
    model = tf.keras.saving.load_model(model_path)
    return model


def main():
    st.title("RecRes App")
    model = load_model()
    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    predictions = []

    if files is not None:
        for file in files:
            # Read the image
            image = Image.open(file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Make the prediction
            prediction = predict(image, model)
            predictions.append(prediction.argmax())

            # Display the prediction
            st.subheader("Prediction:")
            st.write(Classes_reversed[prediction.argmax()])

    # Print all predictions
    st.subheader("All Ingredients:")
    for i, prediction in enumerate(predictions):
        st.write(f"Image {i + 1}: {Classes_reversed[prediction]}")


# Make a prediction
def predict(image, model):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)

    return prediction


if __name__ == "__main__":
    main()
