from model_controller.Model2 import Model2

import os
import numpy as np
import streamlit as st
import pickle
import sqlite3 as sq
import pandas as pd
import nltk
import re
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel

class D2VModel(Model2):
    additional_stop_words = ["advertisement", "advertisements",
                         "cup", "cups",
                         "tablespoon", "tablespoons", 
                         "teaspoon", "teaspoons", 
                         "ounce", "ounces",
                         "salt", 
                         "pepper", 
                         "pound", "pounds",
                         ]

    nltk.download('wordnet')
    nltk.download("stopwords")
    MODEL_PATH = 'saved_models/models/nlp'
    # MODEL_EMBEDDINGS_PATH = os.path.join(MODEL_PATH, 'similarity_embeddings')
    MODEL_EMBEDDINGS_PATH = ('saved_models/models/nlp/similarity_embeddings/')
    CUISINE_CLASSES = ['greek','southern_us','filipino','indian','jamaican','spanish','italian','mexican','chinese','british','thai','vietnamese','cajun_creole','brazilian','french','japanese','irish','korean','moroccan','russian']
    cuisine_model = tf.keras.models.load_model('saved_models/models/nlp/cuisine_model2')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def utils_preprocess_text(self, text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
        ## clean (convert to lowercase and remove punctuations and characters and then strip)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
        ## Tokenize (convert from string to list)
        lst_text = text.split()

        ## remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in 
                        lst_stopwords]
                
        ## Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]
                
        ## Lemmatisation (convert the word into root word)
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]
            
        ## back to string from list
        text = " ".join(lst_text)

        ## Remove digits
        text = ''.join([i for i in text if not i.isdigit()])

        ## remove mutliple space
        text = re.sub(' +', ' ', text)

        return text


    def get_tokenize_text(self, input_text):
        # list of stopwords
        stop_word_list = nltk.corpus.stopwords.words("english")

        # Extend list of stop words
        stop_word_list.extend(self.additional_stop_words)

        return self.utils_preprocess_text(input_text, flg_stemm=False, flg_lemm=True, lst_stopwords=stop_word_list)
    
    def get_df_from_db(self, cuisine):
        try:
            db = sq.connect('recipes.db')
            sql_query = "SELECT title, instructions, ingredients, ingredients_query FROM main_recipes WHERE cuisine = ?"
            return pd.read_sql(sql_query, db, params=(cuisine,))
        except Exception as e:
            print("An error occurred while fetching data from the database:", e)
            return None
        
    
    ## Load from file
    def load_pkl(self, pkl_filename):
        with open(pkl_filename, 'rb') as pkl_file:
            return pickle.load(pkl_file)

    def infer_cuisine_type_on_recipes(self, data):
        model_path = os.path.join(self.MODEL_PATH, 'pickle_model.pkl')
        model = self.load_pkl(model_path)
        data["cuisine"] = model.predict(data["ingredients_query"])
        return data
    
    def predict_cuisine1(self, input_text):
        top = 5
    
        # Tokenize text
        tokenize_text = self.get_tokenize_text(input_text)
    
        # Get model
        model_path = os.path.join(self.MODEL_PATH, 'pickle_model.pkl')
        model = self.load_pkl(model_path)
    
        # Tokenize text
        tokenize_text = self.get_tokenize_text(input_text)

        # Get classes ordered by probability
        proba = model.predict_proba([tokenize_text])[0]

        # Sorted index list 
        indexes = sorted(range(len(proba)), key=lambda k: proba[k], reverse=True)

        # Get cuisine
        cuisine_labels = model.classes_.tolist()
        cusine_ordered = [cuisine_labels[ind] for ind in indexes]

        return cusine_ordered[:top]
    
    def prepare_data(self, input_text, tokenizer):
        token = tokenizer.encode_plus(
            input_text,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        return {
            'input_ids': tf.cast(token.input_ids, tf.float64),
            'attention_mask': tf.cast(token.attention_mask, tf.float64)
        }


    def make_prediction(self ,model, processed_data, classes=['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek', 'indian','irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan','russian', 'southern_us','spanish', 'thai', 'vietnamese'], top_k=5):
        probs = model.predict(processed_data)[0]
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [(classes[i], probs[i]) for i in top_indices]
        return top_predictions

    def get_cuisine(self, ingredients):
        if ingredients is None or len(ingredients) == 0:
            return []
        input_text = ingredients
        processed_data = self.prepare_data(input_text, self.tokenizer)
        results = self.make_prediction(self.cuisine_model, processed_data=processed_data)

        # Extract cuisine names from results
        cuisines = [cuisine for cuisine, _ in results]
        cuisine_string = ''
        count = 1
        for cuisine in cuisines:
            cuisine_string += f"{count}: {cuisine}   "
            count += 1
    

        return cuisine_string
       



    def switch_cuisine(self, cuisine):
        model_paths = {
            'greek': 'saved_models/models/nlp/similarity_embeddings/d2v_greek.pkl',
            'southern_us': 'saved_models/models/nlp/similarity_embeddings/d2v_southern_us.pkl',
            'filipino': 'saved_models/models/nlp/similarity_embeddings/d2v_filipino.pkl',
            'indian': 'saved_models/models/nlp/similarity_embeddings/d2v_indian.pkl',
            'jamaican': 'saved_models/models/nlp/similarity_embeddings/d2v_jamaican.pkl',
            'spanish': 'saved_models/models/nlp/similarity_embeddings/d2v_spanish.pkl',
            'italian': 'saved_models/models/nlp/similarity_embeddings/d2v_italian.pkl',
            'mexican': 'saved_models/models/nlp/similarity_embeddings/d2v_mexican.pkl',
            'chinese': 'saved_models/models/nlp/similarity_embeddings/d2v_chinese.pkl',
            'british': 'saved_models/models/nlp/similarity_embeddings/d2v_british.pkl',
            'thai': 'saved_models/models/nlp/similarity_embeddings/d2v_thai.pkl',
            'vietnamese': 'saved_models/models/nlp/similarity_embeddings/d2v_vietnamese.pkl',
            'cajun_creole': 'saved_models/models/nlp/similarity_embeddings/d2v_cajun_creole.pkl',
            'brazilian': 'saved_models/models/nlp/similarity_embeddings/d2v_brazilian.pkl',
            'french': 'saved_models/models/nlp/similarity_embeddings/d2v_french.pkl',
            'japanese': 'saved_models/models/nlp/similarity_embeddings/d2v_japanese.pkl',
            'irish': 'saved_models/models/nlp/similarity_embeddings/d2v_irish.pkl',
            'korean': 'saved_models/models/nlp/similarity_embeddings/d2v_korean.pkl',
            'moroccan': 'saved_models/models/nlp/similarity_embeddings/d2v_moroccan.pkl',
            'russian': 'saved_models/models/nlp/similarity_embeddings/d2v_russian.pkl',
        }

        if cuisine in model_paths:
            return self.load_pkl(model_paths[cuisine])

        return None



    def get_recipes(self, input_text, cuisine, top_k=5):
        # Tokenize text
        tokenize_text = self.get_tokenize_text(input_text).split()
        try:
            d2v = self.switch_cuisine(cuisine)
        except Exception as e:
            print("An error occurred while switching cuisine:", e)
            return None
        # d2v = self.switch_cuisine(cuisine)
        # Load model from the selected cuisine
        # d2v = self.load_pkl(self.MODEL_EMBEDDINGS_PATH + f'd2v_{cuisine}.pkl')
        
        # Get embeddings
        try:
            embeddings = d2v.infer_vector(tokenize_text)
        except Exception as e:
            print("An error occurred while inferring embeddings:", e)
            return None
        best_recipes = d2v.docvecs.most_similar([embeddings]) #gives you top 10 document tags and their cosine similarity
            
        # Get recipes
        best_recipes_index = [int(output[0]) for output in best_recipes]
    
        # Get dDtaFrame
        df = self.get_df_from_db(cuisine)
        if df is None:
            return None
    
        try:
            filtered_df = df.loc[df.index.isin(best_recipes_index), ["title", "instructions", "ingredients"]]
            filtered_df = filtered_df.reset_index(drop=True)
            return filtered_df.head(top_k)
        except Exception as e:
            print("An error occurred while indexing the DataFrame:", e)
            return None
    
    
    # def get_output(self, ingredients):
    #     if ingredients is None or len(ingredients) == 0:
    #         return []

    #     recipe = self.RecSys(ingredients)

    #     response = []
    #     for index, row in recipe.iterrows():
    #         response.append({
    #             'recipe': str(row['recipe']),
    #             'score': str(row['score']),
    #             'ingredients': str(row['ingredients']),
    #             'url': str(row['url'])
    #         })
    #     return response

    # def format_output(self, ingredients) -> str:
    #     output: list = self.get_output(ingredients)
    #     res = ""
    #     for i in range(len(output)):
    #         recipe_data = output[i]
    #         res += f"""
    #         ### {i + 1}. {recipe_data["recipe"]}
    #         #### Matching Score
    #         {format(float(recipe_data["score"]) * 100, ".2f")}%
    #         #### Ingredients
    #         {recipe_data["ingredients"]}
    #         #### Recipe Link
    #         {recipe_data["url"]}
    #         """
    #     return res

