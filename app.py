import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_attractions(user_query, data, top_n=3):
    """
    Recommend attractions based on user query using TF-IDF and cosine similarity.
    """
    # Combine relevant features for TF-IDF
    data['combined_features'] = data['Description'] + ' ' + data['Activities'] + ' ' + data['Country']
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    
    # Transform user query
    user_vector = tfidf.transform([user_query])
    
    # Calculate similarities
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    
    # Get top recommendations
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    recommendations = data.iloc[top_indices]
    
    return recommendations.iloc[0]['Country'], recommendations

def set_custom_style():
    """Set custom CSS styles for the app."""
    st.markdown("""
        <style>
        /* Custom background with gradient */
        .stApp {
            background: linear-gradient(to bottom right, #f0f8ff, #e6e9ff);
        }
        
        /* Make containers stand out */
        .stMarkdown, .stButton, .stTextArea {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Style for recommendation text */
        .recommendation-text {
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1e90ff;
            font-size: 1.2em;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Style for country name */
        .country-highlight {
            color: #1e90ff;
            font-size: 1.5em;
            font-weight: bold;
            text-decoration: underline;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    # Set custom styling
    set_custom_style()
    
    # Page config
    st.set_page_config(
        page_title="Travel Destination Recommender",
        page_icon="✈️",
        layout="wide"
    )
    
    # Header
    st.title("✈️ Travel Destination Recommender")
    st.write("""
    Describe your desired activities, and we'll recommend the perfect attractions for you!
    Try mentioning specific activities like 'hiking', 'snorkeling', or 'historical sites'.
    """)
    
    # Create sample data
    data = pd.DataFrame({
        'Country': ['Canada', 'Japan', 'Italy', 'New Zealand', 'Thailand'],
        'Attraction': ['Banff National Park', 'Mount Fuji', 'Colosseum', 'Milford Sound', 'Phi Phi Islands'],
        'Description': [
            'Beautiful mountain landscapes with hiking trails and wildlife',
            'Iconic mountain with cultural significance and hiking opportunities',
            'Ancient Roman amphitheater with rich history',
            'Stunning fjord with waterfalls and wildlife viewing',
            'Tropical islands with beaches and snorkeling'
        ],
        'Activities': [
            'hiking, wildlife viewing, photography, camping',
            'hiking, cultural tours, photography, mountain climbing',
            'historical tours, architecture, photography',
            'hiking, kayaking, wildlife viewing, photography',
            'snorkeling, swimming, beach activities, island hopping'
        ]
    })
    
    # User input
    user_query = st.text_area(
        "What kind of activities are you interested in?",
        height=100,
        placeholder="E.g., I want to go hiking in mountains and explore local culture..."
    )
    
    # Add a recommend button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        recommend_button = st.button("🔍 Get Recommendations", use_container_width=True)
    
    # Generate recommendations
    if recommend_button:
        if not user_query:
            st.warning("Please enter your travel preferences to get recommendations.")
            return
            
        try:
            with st.spinner("Finding the perfect destinations for you..."):
                predicted_country, recommendations = recommend_attractions(user_query, data)
                
                # Display results with custom styling
                st.markdown(
                    f'<div class="recommendation-text">Based on your preferences, we recommend visiting: '
                    f'<span class="country-highlight">{predicted_country}</span>!</div>',
                    unsafe_allow_html=True
                )
                
                # Display recommendations in styled cards
                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown("""---""")
                        st.markdown(f"### 🎯 {row['Attraction']}")
                        st.markdown(f"**Location:** {row['Country']}")
                        st.markdown(f"**Description:** {row['Description']}")
                        st.markdown(f"**Activities:** {row['Activities']}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()