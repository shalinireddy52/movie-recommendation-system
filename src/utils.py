def save_recommendations_to_file(recommendations, file_name):
    """Save the list of recommendations to a CSV file."""
    recommendations.to_csv(file_name, header=True)
