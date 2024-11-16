import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_attractions(user_query, pipeline, data, top_n=5):
    """
    Generate travel recommendations based on user query.
    
    Parameters:
    user_query (str): User's input describing desired activities
    pipeline: Trained classification pipeline
    data (pd.DataFrame): Dataset containing attractions
    top_n (int): Number of recommendations to return
    
    Returns:
    tuple: (predicted_country, DataFrame of recommendations) or str if error
    """
    try:
        # Validate input
        if not isinstance(user_query, str) or not user_query.strip():
            raise ValueError("User query must be a non-empty string")
            
        # Predict country
        processed_query = pd.Series([user_query])
        predicted_country = pipeline.predict(processed_query)[0]
        
        # Filter attractions
        attractions_data = data[data['Country'] == predicted_country]
        
        if attractions_data.empty:
            return f"No attractions found for the predicted country: {predicted_country}"
        
        # Calculate similarities
        vectorizer_attractions = TfidfVectorizer(stop_words='english')
        filtered_tfidf = vectorizer_attractions.fit_transform(attractions_data['Description'])
        query_tfidf = vectorizer_attractions.transform([user_query])
        
        # Get similarity scores
        similarity_scores = cosine_similarity(query_tfidf, filtered_tfidf).flatten()
        attractions_data = attractions_data.copy()  # Create copy to avoid SettingWithCopyWarning
        attractions_data['Similarity'] = similarity_scores
        
        # Get top recommendations
        recommendations = (attractions_data
                        .sort_values(by='Similarity', ascending=False)
                        .head(top_n)[['Attraction', 'Country', 'Description']]
                        .copy())
        
        return predicted_country, recommendations
        
    except Exception as e:
        raise Exception(f"Error generating recommendations: {str(e)}")