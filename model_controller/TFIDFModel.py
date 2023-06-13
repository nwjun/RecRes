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

nltk.download("ordnet")


class TFIDFModel(Model):
    # Import datasets
    df = pd.read_csv(os.path.join("saved_models", "tfidf", "tfidf_data.csv"))
    # Import models
    tfidf = joblib.load(os.path.join("saved_models", "tfidf", "tfidf.joblib"))
    tfidf_recipe = joblib.load(os.path.join("saved_models", "tfidf", "tfidf_recipe.joblib"))

    def ingredient_parser(self, ingredients: str | list):
        measures = ["teaspoon", "t", "tsp.", "tablespoon" "T", "tbl.", "tb", "tbsp.", "fluid ounce", "fl oz", "ill",
                    "cup", "c" "pint", "p", "pt", "fl pt", "quart", "q", "qt", "fl qt", "gallon", "g", "gal", "ml",
                    "milliliter", "millilitre", "cc", "mL", "l", "liter", "litre", "L", "dl", "deciliter", "decilitre",
                    "dL", "bulb", "level", "heaped", "rounded", "whole", "pinch", "medium", "slice", "pound", "lb", "#",
                    "ounce", "oz", "mg", "milligram", "milligramme", "g", "gram", "gramme", "kg", "kilogram",
                    "kilogramme", "x", "f", "mm", "millimetre", "millimeter", "cm", "centimeter", "centimetre", "m",
                    "meter", "etre", "inch", "in", "milli", "centi", "deci", "hecto", "kilo"]
        stopwords = ["fresh", "oil", "a", "red", "bunch", "and", "clove", "r", "leaf", "chilli", "large",
                     "extra", "sprig", "ground", "handful", "free" "mall", "pepper", "virgin", "range", "from",
                     "dried", "sustainable", "black", "peeled", "higher", "welfare", "seed", "for", "finely",
                     "freshly", "sea", "quality", "white", "ripe", "few", "piece", "source", "to", "organic",
                     "flat", "smoked", "ginger", "sliced", "green", "picked", "the", "stick", "plain", "plus",
                     "mixed", "mint", "bay", "basil", "your", "cumin", "optional", "fennel", "serve", "mustard",
                     "unsalted", "baby", "paprika", "fat", "ask", "natural", "skin", "oughly", "into", "such",
                     "cut", "good", "brown", "grated", "trimmed", "oregano", "powder", "yellow", "dusting",
                     "nob", "frozen", "on", "deseeded", "low" "runny", "balsamic", "cooked", "streaky",
                     "nutmeg", "sage", "rasher", "zest", "pin", "groundnut", "breadcrumb", "turmeric", "halved",
                     "grating", "stalk", "light", "tinned", "dry", "soft", "rocket", "bone", "colour", "washed",
                     "skinless", "leftover", "splash", "removed", "dijon", "thick", "big", "hot", "drained",
                     "sized", "chestnut", "watercress", "fishmonger", "english", "dill", "caper", "raw",
                     "worcestershire", "flake", "cider", "cayenne", "tbsp", "leg", "pine", "wild", "if", "fine",
                     "herb", "almond", "shoulder", "cube", "dressing", "with", "chunk", "spice", "thumb", "garam",
                     "new", "little", "punnet", "peppercorn", "shelled", "saffron", "other''chopped", "salt",
                     "olive", "taste", "can", "sauce", "water", "diced", "package", "italian", "shredded",
                     "divided", "parsley", "vinegar", "all", "purpose", "crushed", "juice", "more", "coriander",
                     "bell", "needed", "thinly", "boneless", "half", "thyme", "cubed", "cinnamon", "cilantro",
                     "jar", "seasoning", "rosemary", "extract", "sweet", "baking", "beaten", "heavy", "seeded",
                     "tin", "vanilla", "uncooked", "crumb", "style", "thin", "nut", "coarsely", "pring", "chili",
                     "cornstarch", "strip", "cardamom", "rinsed", "honey", "cherry", "root", "quartered", "head",
                     "softened", "container", "crumbled", "frying", "lean", "cooking", "roasted", "warm",
                     "whipping", "thawed", "corn", "pitted", "sun", "kosher", "bite", "toasted", "lasagna",
                     "split", "melted", "degree", "lengthwise", "romano", "packed", "pod" "anchovy", "rom",
                     "prepared", "uiced", "fluid", "floret", "room", "active", "seasoned", "mix", "deveined",
                     "lightly", "anise", "thai", "size", "unsweetened", "torn", "wedge", "sour", "basmati",
                     "marinara", "dark", "temperature", "garnish", "bouillon", "loaf", "shell", "reggiano",
                     "canola", "parmigiano", "ound", "canned", "ghee", "crust", "long" "broken", "ketchup",
                     "bulk", "cleaned", "condensed", "sherry", "provolone", "cold", "soda", "cottage", "spray",
                     "tamarind", "pecorino", "shortening" "part", "bottle", "sodium", "cocoa", "grain", "french",
                     "roast", "stem", "link", "firm" "asafoetida", "mild", "dash", "boiling"]
        # Convert ingredients string into a list
        if isinstance(ingredients, str):
            ingredients = re.split(r",\s*", ingredients)

        parsed_ingredients = []
        for i in ingredients:
            # Remove punctuations
            translator = str.maketrans('', '', string.punctuation)
            i.translate(translator)
            # Split up with hyphens and spaces
            items = re.split("[ -]", i)
            # Remove non-alphabetic strings
            items = [word for word in items if word.isalpha()]
            # Convert to lowercase
            items = [word.lower() for word in items]
            # Remove accents
            items = [unidecode.unidecode(word) for word in items]
            # Lemmatize words
            lemmatizer = WordNetLemmatizer()
            items = [lemmatizer.lemmatize(word) for word in items]
            # Remove stopwords
            items = [word for word in items if word not in measures]
            items = [word for word in items if word not in stopwords]
            if items:
                parsed_ingredients.append(" ".join(items))
        parsed_ingredients = " ".join(parsed_ingredients)
        return parsed_ingredients

    def ingredient_parser_final(self, ingredients):
        if not isinstance(ingredients, list):
            ingredients = ast.literal_eval(ingredients)
        ingredients = ",".join(ingredients)
        ingredients = unidecode.unidecode(ingredients)
        return ingredients

    def title_parser(self, title):
        title = unidecode.unidecode(title)
        return title

    def get_recommendations(self, ingredients, n=3):
        ingredients_parsed = self.ingredient_parser(ingredients)
        ingredients_tfidf = self.tfidf.transform([ingredients_parsed])

        # Calculate cosine similarities between actual and test recipe ingredients
        scores = list([cosine_similarity(ingredients_tfidf, x) for x in self.tfidf_recipe])

        # Get the highest n scores
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        # Create dataframe to load in recommendations
        recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score", "url"])
        count = 0
        for i in top:
            recommendation.at[count, "recipe"] = self.title_parser(self.df["recipe_name"][i])
            recommendation.at[count, "ingredients"] = self.ingredient_parser_final(self.df["ingredients"][i])
            recommendation.at[count, "url"] = self.df["recipe_urls"][i]
            recommendation.at[count, "score"] = "{:.3f}".format(float(scores[i]))
            count += 1
        return recommendation

    def get_output(self, ingredients):
        if ingredients is None or len(ingredients) == 0:
            return []

        recipe = self.get_recommendations(ingredients)

        response = []
        for index, row in recipe.iterrows():
            response.append({
                "recipe": row["recipe"],
                "score": row["score"],
                "ingredients": row["ingredients"],
                "url": row["url"]
            })
        return response

    def format_output(self, ingredients) -> str:
        output: list = self.get_output(ingredients)
        regex = r",(?!\s)"
        newline = "\n"
        res = ""
        for i in range(len(output)):
            recipe_data = output[i]
            print(recipe_data)
            res += f"""
### {i + 1}. {recipe_data["recipe"]}
#### Matching Score
{format(float(recipe_data["score"]) * 100, ".2f")}%
#### Ingredients
{newline.join([f"- {i}" for i in re.split(regex, recipe_data["ingredients"])])}
#### Recipe Link
{recipe_data["url"]}
"""
        return res
