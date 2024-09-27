import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def recommend_movies(user_data, movies_data):
    # Combine watched genres and favorite actors into a single preference column
    user_data['Preferences'] = user_data['Watched Genres'] + ' ' + user_data['Favorite Actors']
    
    # Vectorize the combined preferences
    vectorizer = CountVectorizer()
    user_vectors = vectorizer.fit_transform(user_data['Preferences'])
    
    # Compute similarity using cosine similarity
    similarity_matrix = cosine_similarity(user_vectors, user_vectors)
    
    # Assume current user is the last one in the dataset
    user_index = user_data.index[-1]
    
    # Find most similar users
    similar_users = list(enumerate(similarity_matrix[user_index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]  # Top 5
    
    recommendations = []
    for i in similar_users:
        user_movies = movies_data[movies_data['User_ID'] == i[0]]['Movie_Name']
        recommendations.extend(user_movies)
    
    return list(set(recommendations))
