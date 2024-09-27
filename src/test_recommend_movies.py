import unittest
from recommend_movies import recommend_movies  # Import your function

class TestRecommendMovies(unittest.TestCase):

    def test_recommendation_by_genres(self):
        # Test case for recommendation based on genres
        user_data = {'Watched Genres': 'Action', 'Favorite Actors': 'Actor A'}
        movie_data = {}  # Provide mock movie data here
        recommended_movies = recommend_movies(user_data, movie_data)
        self.assertIn('Action Movie', recommended_movies)

    def test_recommendation_by_actors(self):
        # Test case for recommendation based on actors
        user_data = {'Watched Genres': 'Comedy', 'Favorite Actors': 'Actor B'}
        movie_data = {}  # Provide mock movie data here
        recommended_movies = recommend_movies(user_data, movie_data)
        self.assertIn('Comedy Movie with Actor B', recommended_movies)

if __name__ == '__main__':
    unittest.main()
