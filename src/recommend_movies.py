import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy


# Content-Based Filtering function
def content_based_filtering(user_data, movie_data):
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
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 similar users

    # Recommend movies watched by similar users
    recommendations = []
    for i in similar_users:
        user_movies = movie_data[movie_data['User_ID'] == i[0]]['Movie_Name']
        recommendations.extend(user_movies)

    return list(set(recommendations))

# Collaborative Filtering function
def collaborative_filtering(user_data, movie_data):
    # Prepare the data for collaborative filtering (ratings-based)
    reader = Reader(rating_scale=(1, 5))
    ratings_data = movie_data[['User_ID', 'Movie_Name', 'Ratings']]

    # Load the dataset into Surprise
    data = Dataset.load_from_df(ratings_data, reader)

    # Split data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.25)

    # Use KNN (user-based collaborative filtering)
    algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)

    # Predict for the current user (assumed as the first one in the list)
    user_id = user_data['User_ID'][0]
    user_ratings = movie_data[movie_data['User_ID'] == user_id]

    recommended_movies = []
    for movie in movie_data['Movie_Name'].unique():
        if movie not in user_ratings['Movie_Name'].values:
            predicted_rating = algo.predict(user_id, movie).est
            if predicted_rating >= 4:  # Recommend movies with predicted rating >= 4
                recommended_movies.append(movie)

    return recommended_movies


# Hybrid Recommendation Function: Combine content-based and collaborative filtering
def recommend_movies(user_data, movie_data):
    # Apply content-based filtering
    content_recommendations = content_based_filtering(user_data, movie_data)

    # Apply collaborative filtering
    collaborative_recommendations = collaborative_filtering(user_data, movie_data)

    # Combine both recommendations (Hybrid approach)
    final_recommendations = list(set(content_recommendations + collaborative_recommendations))

    return final_recommendations