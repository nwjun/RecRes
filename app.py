import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_option_menu import option_menu
from st_on_hover_tabs import on_hover_tabs
from os import path as osp

from model_controller.Model import Model
from model_controller.TFIDFModel import TFIDFModel

st.set_page_config(layout="wide")

INGREDIENT_CLS_CLASSES = [
    "Zucchini",
    "Yoghurt",
    "Tomato",
    "Soyghurt",
    "Sour-Cream",
    "Satsumas",
    "Red-Grapfruit",
    "Red-Beet",
    "Potato",
    "Pomegranate",
    "Pineapple",
    "Pepper",
    "Pear",
    "Peach",
    "Passion-Fruit",
    "Orange",
    "Onion",
    "Oatghurt",
    "Oat-Milk",
    "Milk",
    "Melon",
    "Mango",
    "Lime",
    "Lemon",
    "Leek",
    "Kiwi",
    "Juice",
    "Ginger",
    "Cucumber",
    "Carrots",
    "Cabbage",
    "Brown-Cap-Mushroom",
    "Banana",
    "Avocado",
    "Aubergine",
    "Asparagus",
    "Apple",
]

# Reverse the Classes list
INGREDIENT_CLS_CLASSES = INGREDIENT_CLS_CLASSES[::-1]


# Load the model
@st.cache_resource
def load_ingredient_cls_model():
    model = None
    # model_path = osp.join(
    #     "saved_models", "groceries-image-recognition", "resnet50_frozen_model"
    # )
    # model = tf.keras.saving.load_model(model_path)
    return model


# demo code for loading recipe model
@st.cache_resource
def load_recipe_model_demo():
    model = None
    return model


def main():
    st.title("RecRes App")

    with st.sidebar:
        tabs = on_hover_tabs(
            tabName=["Recipe", "Restaurant"],
            iconName=["soup_kitchen", "restaurant"],
            default_choice=0,
        )

    if tabs.lower() == "recipe":
        recipe_recommender_page()

    elif tabs.lower() == "restaurant":
        st.write("## sss")


def recipe_recommender_page():
    # TODO: Add other model
    RECIPE_MODEL_FAC = {
        "TFIDF Vectorizer": TFIDFModel(),
        "Attention Encoder-Decoder": load_recipe_model_demo,
    }

    st.write("## Recipe Recommender")
    ingredient_cls_model = load_ingredient_cls_model()
    st.write("### 1. What ingredients do you have?")
    st.write("#### Lazy to type? Just take a picture of them!")
    st.info(f'Currently only able to detect {", ".join(INGREDIENT_CLS_CLASSES)}')
    files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg"], accept_multiple_files=True
    )
    ingredients = []
    images = []
    num_cols = 5

    if files is not None:
        with st.spinner("Loading..."):
            for file in files:
                # Read the image
                image = Image.open(file)
                images.append(image)

                # Make the prediction
                prediction = ingredient_cls_predict(image, ingredient_cls_model)
                ingredients.append(INGREDIENT_CLS_CLASSES[prediction.argmax()])

        num_rows = max(
            ((len(files) + 1) // num_cols) if num_cols != 1 else len(files), 1
        )
        for r in range(num_rows):
            cols = st.columns(num_cols)
            for c in range(num_cols):
                idx = r * num_cols + c
                if idx >= len(files):
                    break
                cols[c].image(images[idx], width=150, caption=ingredients[idx])

    st.write('#### Ops, detection failed? You can add them manually ><"')
    text_inp_ingredients = st.text_input(
        label="Ingredient(s), separate them by ', ' (with space). E.g: Apple, orange"
    )
    text_inp_ingredients = text_inp_ingredients.split(", ")
    ingredients.extend(text_inp_ingredients)
    st.write(f'Your ingredient(s): {", ".join(ingredients)}')

    st.write("### 2. Get your personalized recipe!")
    recipe_arch = option_menu(
        None,
        ["TFIDF Vectorizer", "Attention Encoder-Decoder"],
        icons=["0-square", "1-square"],
        default_index=0,
        orientation="horizontal",
    )

    recipe_model: Model = RECIPE_MODEL_FAC[recipe_arch]
    if recipe_arch == "TFIDF Vectorizer":
        st.write(recipe_model.format_output(", ".join(ingredients)))
    elif recipe_arch == "Attention Encoder-Decoder":
        calorie = st.selectbox('#### What calorie level are you currently aiming for?',('<Select>','Low', 'Medium', 'High'))
        if calorie == '<Select>':
            st.error('Please select a calorie level.')
        food = st.text_input('#### What food are you craving for now?', placeholder='Big Mac Pizza')
        ingredient = st.text_input('#### Please let me know what ingredients you have now.', placeholder='Beef, Thousand Island, Cheese')

        calorie_mapping = {
            '<Select>': None,
            'Low': 0,
            'Medium': 1,
            'High': 2
        }
        calorie_value = calorie_mapping[calorie]
        food_value = food.strip()
        ingredient_value = [item.strip() for item in ingredient.split(",")]

        print(calorie_value)
        print(food_value)
        print(ingredient_value)

        personalized_recipe = st.button('Find Out Now!')

        if (personalized_recipe):
            attentionModel


# Make a prediction
def ingredient_cls_predict(image, model):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)

    return prediction


def attentionModel(calorie, food, ingredient):
    return calorie, food, ingredient


if __name__ == "__main__":
    st.markdown(
        "<style>" + open("./style.scss").read() + "</style>", unsafe_allow_html=True
    )
    main()
