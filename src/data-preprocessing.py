import pandas as pd

def load_movies_data(file_path):
    """Load movies data from CSV file."""
    return pd.read_csv(file_path)

def load_ratings_data(file_path):
    """Load ratings data from CSV file."""
    return pd.read_csv(file_path)

def merge_data(movies, ratings):
    """Merge movies and ratings data."""
    return pd.merge(ratings, movies, on='movieId')
