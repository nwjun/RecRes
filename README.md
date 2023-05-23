# Recipe and Restaurant Recommendation System (RecRes)

## Related Datasets

### Recipe

|Dataset|Features|#Data |Remark|Used|
|-------|--------|:----:|------|:---:|
|[Food Recommendation System - schemersays](https://www.kaggle.com/datasets/schemersays/food-recommendation-system?select=1662574418893344.csv)|name, ingredients, cuisines, ratings|400|||
|[Food.com - Recipes](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)|name, preparing time, date, tags, nutrition, # cooking steps, description, ingredients, #ingredients|180K|Provides raw and tokenize data. [Paper](https://aclanthology.org/D19-1613/).|
|[Food.com - Review of Recipes](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)|date, rating|700K|Provides raw and tokenize data. [Paper](https://aclanthology.org/D19-1613/).||
|[Indian Food and Its Recipes Dataset (With Images)](https://www.kaggle.com/datasets/kishanpahadiya/indian-food-and-its-recipes-dataset-with-images)|name, image_url, description, cuisine, course, diet, prep_time, ingredients, instructions| 4226|Scraped from [Archana's Kitchen](https://www.archanaskitchen.com/)||
|[MealRec](https://github.com/WUT-IDEA/MealRec)|name, #reviews, category, aver_rate, image_url, ingredients, cooking_directions, nutritions, reviews, tags|7280|There are multiple reviews for one recipe|
|[Indian Food 101](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101)|name, ingredients, diet, prep_time, cook_time, flavor_profile, course, state, region|255|
|[foodRecSys-V1](https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1?select=core-data-valid_rating.csv)|recipe_name, image_url, ingredients, cooking_directions, nutritions, rating|45568|
### Restaurant

|Dataset|Features|#Data |Remark|Used|
|-------|--------|:----:|------|:---:|
|[Restaurant Data with Consumer Ratings](https://www.kaggle.com/datasets/uciml/restaurant-data-with-consumer-ratings)|payment type, operating hours, operating days, parking_lot, latitude, longitude, the_geom_meter, name, address, city, state, country, fax, zip, alchohol, smoking_area, dress_code, accessibility, price, url, Rambience, franchise, area, other_services, rating, food_rating, service_rating|-|Consist of multiple csv of users and restaurants with different length|
|[Micheline Guide Restaurants](https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021)|name, address, location, price, cuisine, longitude, latitude, phoneNumber, Url, WebsiteUrl|6653|


## Related Tutorials and Repositories

|Name|Description|Remark|
|---|---|---|
|[Scraping Google Reviews with Selenium(Python)](https://medium.com/@isguzarsezgin/scraping-google-reviews-with-selenium-python-23135ffcc331)|Web scraping google reviews via Selenium and BeautifulSoup|
|[recipe-recommendation-system](https://github.com/ajemerson/recipe-recommendation-system)|Data-driven recipe recommendation system using web-scraped recipe data (including but not limited to data like ingredients, health facts, etc.) and user’s historical preference|No access to dataset
|[recipes-telegram-bot](https://github.com/RomainGratier/recipes-telegram-bot)|Telegram bot that can recommend recipes based on the ingredients you already have|Technical article on [Medium](https://romain-gratier.medium.com/de2d314f565d?source=friends_link&sk=c5280f8c50aa5551d1b36619891e9b4f)|
|[Food Recommendation using BERT](https://www.kaggle.com/code/ajitrajput/food-recommendation-using-bert/input)|Based on cosine similarities computed on embedding from BERT|
|[Restaurant_recommendation_system](https://github.com/MariloyH/Restaurant_recommendation_system)|Recommend users based on ratings and comments of other visitors taking into consideration of their location and preferences|[Slides](https://docs.google.com/presentation/d/1ZlSZUL6SJBcRnLjmMwqcynuWotso9JrDRmxAZ9-IRTA/edit#slide=id.p1/google_docs)