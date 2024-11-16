import streamlit as st
import pandas as pd
import pickle
from recommendation import recommend_attractions
from custom_preprocessors import PreprocessText
import os
import subprocess

# Path to the setup script
setup_script = "./setup.sh"

# Run setup.sh if it exists
if os.path.exists(setup_script):
    print("Running setup.sh...")
    try:
        subprocess.run(["bash", setup_script], check=True)
        print("setup.sh executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running setup.sh: {e}")
else:
    print("setup.sh not found, skipping setup.")
def load_resources():
    """Load the pipeline and dataset."""
    try:
        pipeline = pickle.load(open("destination_pipeline.pkl", "rb"))
        data = pd.read_csv("best_travel_destinations_for_2025.csv")
        return pipeline, data
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None

def main():
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
    
    # Load resources
    pipeline, data = load_resources()
    if not pipeline or data is None:
        st.stop()
    
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
                result = recommend_attractions(user_query, pipeline, data)
                
            if isinstance(result, str):
                st.warning(result)
            else:
                predicted_country, recommendations = result
                
                # Display results in a nice format
                st.success(f"Based on your preferences, we recommend visiting: **{predicted_country}**!")
                
                # Display recommendations in cards
                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown("""---""")
                        st.markdown(f"### 🎯 {row['Attraction']}")
                        st.markdown(f"**Location:** {row['Country']}")
                        st.markdown(f"**Description:** {row['Description']}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()