from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(user_movie_matrix):
    """Compute cosine similarity between movies."""
    return cosine_similarity(user_movie_matrix.T)

def recommend_movies(movie_name, cosine_sim_df, top_n=5):
    """Recommend top N similar movies based on cosine similarity."""
    similar_scores = cosine_sim_df[movie_name]
    similar_movies = similar_scores.sort_values(ascending=False)
    similar_movies = similar_movies.drop(movie_name)
    return similar_movies.head(top_n)
