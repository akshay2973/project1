








import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset

data = pd.read_csv('Dataset .csv')

# Preprocessing
data['Cuisines'] = data['Cuisines'].fillna('Unknown')
data['Features'] = (
    data['Cuisines'] + ' ' +
    data['Price range'].astype(str) + ' ' +
    data['Aggregate rating'].astype(str)
)
vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(data['Features'])
similarity_matrix = cosine_similarity(feature_matrix)


def recommend_restaurants(user_cuisine, user_price_range, user_rating, top_n=5):
    user_input = f"{user_cuisine} {user_price_range} {user_rating}"
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()
    
    recommendations = data.copy()
    recommendations['Similarity'] = similarity_scores
    recommendations = recommendations[
        (recommendations['Aggregate rating'] >= user_rating) &
        (recommendations['Price range'] == user_price_range)
    ]
    recommendations = recommendations.sort_values(by='Similarity', ascending=False)
    
    return recommendations[['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating']].head(top_n)

# Streamlit Web Application
st.title("Restaurant Recommendation System")

st.sidebar.header("Enter Your Preferences")
user_cuisine = st.sidebar.text_input("Cuisine (e.g., Japanese, Italian):", "Japanese")
user_price_range = st.sidebar.slider("Price Range (1-4):", 1, 4, 3)
user_rating = st.sidebar.slider("Minimum Rating (0.0 - 5.0):", 0.0, 5.0, 4.5)

if st.sidebar.button("Recommend"):
    recommendations = recommend_restaurants(
        user_cuisine=user_cuisine,
        user_price_range=user_price_range,
        user_rating=user_rating,
        top_n=5
    )
    if recommendations.empty:
        st.warning("No matching restaurants found. Try changing your preferences.")
    else:
        st.subheader("Top Recommendations")
        st.table(recommendations)







