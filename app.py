import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_option_menu import option_menu
from st_on_hover_tabs import on_hover_tabs

st.set_page_config(layout='wide')

INGREDIENT_CLS_CLASSES = ['Zucchini', 'Yoghurt', 'Tomato', 'Soyghurt', 'Sour-Cream', 'Satsumas', 'Red-Grapfruit', 'Red-Beet',
           'Potato', 'Pomegranate', 'Pineapple', 'Pepper', 'Pear', 'Peach', 'Passion-Fruit', 'Orange', 'Onion',
           'Oatghurt', 'Oat-Milk', 'Milk', 'Melon', 'Mango', 'Lime', 'Lemon', 'Leek', 'Kiwi', 'Juice', 'Ginger',
           'Cucumber', 'Carrots', 'Cabbage', 'Brown-Cap-Mushroom', 'Banana', 'Avocado', 'Aubergine', 'Asparagus',
           'Apple']

# Reverse the Classes list
INGREDIENT_CLS_CLASSES = INGREDIENT_CLS_CLASSES[::-1]


# Load the model
@st.cache_resource
def load_ingredient_cls_model():
    model_path = "RecipeRecommender/groceries-image-recognition/resnet50_frozen_model"
    model = tf.keras.saving.load_model(model_path)
    return model


# demo code for loading recipe model
@st.cache_resource
def load_recipe_model_demo():
    model = None
    return model


def main():
    st.title("RecRes App")
    
    with st.sidebar:
        tabs = on_hover_tabs(tabName=['Recipe', 'Restaurant'], 
                             iconName=['cooking', 'money'], default_choice=0)
        
    if tabs.lower() == 'recipe':
        recipe_recommender_page()
        
    elif tabs.lower() == 'restaurant':
        st.write("## sss")
        
def recipe_recommender_page():
    # TODO: Add other model
    RECIPE_MODEL_FAC = {
        'model1': load_recipe_model_demo,
        'model2': load_recipe_model_demo
    }
    
    st.write("## Recipe Recommender")
    ingredient_cls_model = load_ingredient_cls_model()
    st.write('### 1. What ingredients do you have?')
    st.write('#### Lazy to type? Just take a picture of them!')
    st.info(f'Currently only able to detect {", ".join(INGREDIENT_CLS_CLASSES)}')
    files = st.file_uploader("Upload images", type=["jpg", "jpeg"], accept_multiple_files=True)
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
            
        num_rows = max(((len(files) + 1) // num_cols) if num_cols != 1 else len(files), 1)
        for r in range(num_rows):
            cols = st.columns(num_cols)
            for c in range(num_cols):
                idx = r*num_cols+c
                if idx >= len(files): break
                cols[c].image(images[idx], width=150, caption=ingredients[idx])
                
    st.write('#### Ops, detection failed? You can add them manually ><"')
    text_inp_ingredients = st.text_input(label="Ingredient(s), separate them by ', ' (with space). E.g: Apple, orange")
    text_inp_ingredients = text_inp_ingredients.split(', ')
    ingredients.extend(text_inp_ingredients)
    st.write(f'Your ingredient(s): {", ".join(ingredients)}')
        
    st.write('### 2. Get your personalized recipe!')
    recipe_arch = option_menu(None, ["model1", "model2"],
                icons=['0-square', '1-square'], default_index=0,
                orientation='horizontal')
    
    if recipe_arch == 'demo':
        st.write('model1 is used')
    elif recipe_arch == 'model2':
        st.write('model2 is used')
        
    recipe_model = RECIPE_MODEL_FAC[recipe_arch]
    # TODO: present recipe


# Make a prediction
def ingredient_cls_predict(image, model):
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)

    return prediction


if __name__ == "__main__":
    main()
