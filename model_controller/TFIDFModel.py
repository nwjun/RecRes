from model_controller.Model import Model

import ast
import os
import re
import string

import pandas as pd
import joblib
import nltk
import unidecode as unidecode
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')


class TFIDFModel(Model):
    df = pd.read_csv(os.path.join("saved_models", "tfidf", "tfidf_data.csv"))
    tfidf_encodings = joblib.load(os.path.join("saved_models", "tfidf", "tfidf_recipe.joblib"))
    tfidf = joblib.load(os.path.join("saved_models", "tfidf", "tfidf.joblib"))

    def ingredient_parser(self, ingredients):
        measures = ['teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'tbl.', 'tb', 'tbsp.', 'fluid ounce', 'fl oz', 'gill',
                    'cup', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'ml',
                    'milliliter', 'millilitre', 'cc', 'mL', 'l', 'liter', 'litre', 'L', 'dl', 'deciliter', 'decilitre',
                    'dL', 'bulb', 'level', 'heaped', 'rounded', 'whole', 'pinch', 'medium', 'slice', 'pound', 'lb', '#',
                    'ounce', 'oz', 'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram',
                    'kilogramme', 'x', 'of', 'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre', 'm',
                    'meter', 'metre', 'inch', 'in', 'milli', 'centi', 'deci', 'hecto', 'kilo']
        words_to_remove = ['fresh', 'oil', 'a', 'red', 'bunch', 'and', 'clove', 'or', 'leaf', 'chilli', 'large',
                           'extra', 'sprig', 'ground', 'handful', 'free', 'small', 'pepper', 'virgin', 'range', 'from',
                           'dried', 'sustainable', 'black', 'peeled', 'higher', 'welfare', 'seed', 'for', 'finely',
                           'freshly', 'sea', 'quality', 'white', 'ripe', 'few', 'piece', 'source', 'to', 'organic',
                           'flat', 'smoked', 'ginger', 'sliced', 'green', 'picked', 'the', 'stick', 'plain', 'plus',
                           'mixed', 'mint', 'bay', 'basil', 'your', 'cumin', 'optional', 'fennel', 'serve', 'mustard',
                           'unsalted', 'baby', 'paprika', 'fat', 'ask', 'natural', 'skin', 'roughly', 'into', 'such',
                           'cut', 'good', 'brown', 'grated', 'trimmed', 'oregano', 'powder', 'yellow', 'dusting',
                           'knob', 'frozen', 'on', 'deseeded', 'low', 'runny', 'balsamic', 'cooked', 'streaky',
                           'nutmeg', 'sage', 'rasher', 'zest', 'pin', 'groundnut', 'breadcrumb', 'turmeric', 'halved',
                           'grating', 'stalk', 'light', 'tinned', 'dry', 'soft', 'rocket', 'bone', 'colour', 'washed',
                           'skinless', 'leftover', 'splash', 'removed', 'dijon', 'thick', 'big', 'hot', 'drained',
                           'sized', 'chestnut', 'watercress', 'fishmonger', 'english', 'dill', 'caper', 'raw',
                           'worcestershire', 'flake', 'cider', 'cayenne', 'tbsp', 'leg', 'pine', 'wild', 'if', 'fine',
                           'herb', 'almond', 'shoulder', 'cube', 'dressing', 'with', 'chunk', 'spice', 'thumb', 'garam',
                           'new', 'little', 'punnet', 'peppercorn', 'shelled', 'saffron', 'other''chopped', 'salt',
                           'olive', 'taste', 'can', 'sauce', 'water', 'diced', 'package', 'italian', 'shredded',
                           'divided', 'parsley', 'vinegar', 'all', 'purpose', 'crushed', 'juice', 'more', 'coriander',
                           'bell', 'needed', 'thinly', 'boneless', 'half', 'thyme', 'cubed', 'cinnamon', 'cilantro',
                           'jar', 'seasoning', 'rosemary', 'extract', 'sweet', 'baking', 'beaten', 'heavy', 'seeded',
                           'tin', 'vanilla', 'uncooked', 'crumb', 'style', 'thin', 'nut', 'coarsely', 'spring', 'chili',
                           'cornstarch', 'strip', 'cardamom', 'rinsed', 'honey', 'cherry', 'root', 'quartered', 'head',
                           'softened', 'container', 'crumbled', 'frying', 'lean', 'cooking', 'roasted', 'warm',
                           'whipping', 'thawed', 'corn', 'pitted', 'sun', 'kosher', 'bite', 'toasted', 'lasagna',
                           'split', 'melted', 'degree', 'lengthwise', 'romano', 'packed', 'pod', 'anchovy', 'rom',
                           'prepared', 'juiced', 'fluid', 'floret', 'room', 'active', 'seasoned', 'mix', 'deveined',
                           'lightly', 'anise', 'thai', 'size', 'unsweetened', 'torn', 'wedge', 'sour', 'basmati',
                           'marinara', 'dark', 'temperature', 'garnish', 'bouillon', 'loaf', 'shell', 'reggiano',
                           'canola', 'parmigiano', 'round', 'canned', 'ghee', 'crust', 'long', 'broken', 'ketchup',
                           'bulk', 'cleaned', 'condensed', 'sherry', 'provolone', 'cold', 'soda', 'cottage', 'spray',
                           'tamarind', 'pecorino', 'shortening', 'part', 'bottle', 'sodium', 'cocoa', 'grain', 'french',
                           'roast', 'stem', 'link', 'firm', 'asafoetida', 'mild', 'dash', 'boiling']
        # The ingredient list is now a string so we need to turn it back into a list. We use ast.literal_eval
        if isinstance(ingredients, list):
            ingredients = ingredients
        else:
            ingredients = ast.literal_eval(ingredients)
        # We first get rid of all the punctuation. We make use of str.maketrans. It takes three input
        # arguments 'x', 'y', 'z'. 'x' and 'y' must be equal-length strings and characters in 'x'
        # are replaced by characters in 'y'. 'z' is a string (string.punctuation here) where each character
        #  in the string is mapped to None.
        translator = str.maketrans('', '', string.punctuation)
        lemmatizer = WordNetLemmatizer()
        ingred_list = []
        for i in ingredients:
            i.translate(translator)
            # We split up with hyphens as well as spaces
            items = re.split(' |-', i)
            # Get rid of words containing non alphabet letters
            items = [word for word in items if word.isalpha()]
            # Turn everything to lowercase
            items = [word.lower() for word in items]
            # remove accents
            items = [unidecode.unidecode(word) for word in
                     items]  # ''.join((c for c in unicodedata.normalize('NFD', items) if unicodedata.category(c) != 'Mn'))
            # Lemmatize words so we can compare words to measuring words
            items = [lemmatizer.lemmatize(word) for word in items]
            # Gets rid of measuring words/phrases, e.g. heaped teaspoon
            items = [word for word in items if word not in measures]
            # Get rid of common easy words
            items = [word for word in items if word not in words_to_remove]
            if items:
                ingred_list.append(' '.join(items))
        ingred_list = " ".join(ingred_list)
        return ingred_list

    # Top-N recomendations order by score
    def get_recommendations(self, n, scores):
        # order the scores with and filter to get the highest N scores
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        # create dataframe to load in recommendations
        recommendation = pd.DataFrame(columns=['recipe', 'ingredients', 'score', 'url'])
        count = 0
        for i in top:
            recommendation.at[count, 'recipe'] = self.title_parser(self.df['recipe_name'][i])
            recommendation.at[count, 'ingredients'] = self.ingredient_parser_final(self.df['ingredients'][i])
            recommendation.at[count, 'url'] = self.df['recipe_urls'][i]
            recommendation.at[count, 'score'] = "{:.3f}".format(float(scores[i]))
            count += 1
        return recommendation

    # neaten the ingredients being outputted
    def ingredient_parser_final(self, ingredient):
        if isinstance(ingredient, list):
            ingredients = ingredient
        else:
            ingredients = ast.literal_eval(ingredient)

        ingredients = ','.join(ingredients)
        ingredients = unidecode.unidecode(ingredients)
        return ingredients

    def title_parser(self, title):
        title = unidecode.unidecode(title)
        return title

    def RecSys(self, ingredients, n=5):
        # parse the ingredients using my ingredient_parser
        try:
            ingredients_parsed = self.ingredient_parser(ingredients)
        except Exception:
            ingredients_parsed = self.ingredient_parser([ingredients])

        # use our pretrained tfidf model to encode our input ingredients
        ingredients_tfidf = self.tfidf.transform([ingredients_parsed])

        # calculate cosine similarity between actual recipe ingreds and test ingreds
        cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), self.tfidf_encodings)
        scores = list(cos_sim)

        # Filter top N recommendations
        recommendations = self.get_recommendations(n, scores)
        return recommendations

    def get_output(self, ingredients):
        if ingredients is None or len(ingredients) == 0:
            return []

        recipe = self.RecSys(ingredients)

        response = []
        for index, row in recipe.iterrows():
            response.append({
                'recipe': str(row['recipe']),
                'score': str(row['score']),
                'ingredients': str(row['ingredients']),
                'url': str(row['url'])
            })
        return response

    def format_output(self, ingredients) -> str:
        output: list = self.get_output(ingredients)
        res = ""
        for i in range(len(output)):
            recipe_data = output[i]
            res += f"""
            ### {i + 1}. {recipe_data["recipe"]}
            #### Matching Score
            {format(float(recipe_data["score"]) * 100, ".2f")}%
            #### Ingredients
            {recipe_data["ingredients"]}
            #### Recipe Link
            {recipe_data["url"]}
            """
        return res
