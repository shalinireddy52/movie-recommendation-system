import pandas as pd
from src.data_preprocessing import load_movies_data, load_ratings_data, merge_data
from src.recommendation_system import compute_cosine_similarity, recommend_movies
from src.utils import save_recommendations_to_file


def main():
    # Load data
    movies = load_movies_data('data/movies.csv')
    ratings = load_ratings_data('data/ratings.csv')

    # Merge data
    movie_data = merge_data(movies, ratings)

    # Create user-movie matrix
    user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(user_movie_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    # Get movie recommendations
    movie_name = 'The Dark Knight'  # Example movie
    recommended_movies = recommend_movies(movie_name, cosine_sim_df, top_n=5)

    # Save recommendations to file
    save_recommendations_to_file(recommended_movies, 'recommended_movies.csv')

    print(f"Top 5 recommended movies for '{movie_name}':")
    print(recommended_movies)


if __name__ == '__main__':
    main()
