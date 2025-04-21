# Project Name: Personalized Medicine Recommending System

## Dhrupad Chakraborty
import numpy as np
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore")

df = pd.read_csv('medicine.csv')
df.head()
df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.duplicated().sum()
df['Description']
df['Description'].apply(lambda x: x.split())
df['Reason'] = df['Reason'].apply(lambda x: x.split())

df['Description'] = df['Description'].apply(lambda x: x.split())
df['Description'] = df['Description'].apply(lambda x: [i.replace(" ", "") for i in x])
df['tags'] = df['Description'] + df['Reason']
new_df = df[['index', 'Drug_Name', 'tags']]
new_df
new_df['tags'].apply(lambda x: " ".join(x))
new_df
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', max_features=5000)


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new_df['tags'] = new_df['tags'].apply(stem)
cv.fit_transform(new_df['tags']).toarray().shape
vectors = cv.fit_transform(new_df['tags']).toarray()
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(vectors)
similarity = cosine_similarity(vectors)
similarity[1]


def recommend(medicine):
    medicine_index = new_df[new_df['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in medicines_list:
        print(new_df.iloc[i[0]].Drug_Name)


recommend("Paracetamol 125mg Syrup 60mlParacetamol 500mg Tablet 10'S")

import pickle

def recommend_by_symptoms(symptoms):
    """
    Recommend medicines based on a list of symptoms.
    Args:
        symptoms (list of str): List of symptoms input by the user.
    Returns:
        list of str: List of recommended medicine names.
    """
    symptoms = [symptom.lower() for symptom in symptoms]
    recommended = []

    for idx, row in new_df.iterrows():
        tags = row['tags'].lower()
        # Count how many symptoms match the tags
        match_count = sum(symptom in tags for symptom in symptoms)
        if match_count > 0:
            recommended.append((row['Drug_Name'], match_count))

    # Sort by number of matching symptoms descending
    recommended = sorted(recommended, key=lambda x: x[1], reverse=True)
    # Return top 5 medicine names
    return [med[0] for med in recommended[:5]]

pickle.dump(new_df.to_dict(), open('medicine_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
