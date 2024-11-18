import streamlit as st
import pandas as pd
import pickle
from recommendation import recommend_attractions
from custom_preprocessors import PreprocessText

def set_custom_style():
    st.markdown("""
        <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Main background with parallax effect */
        .stApp {
            background: linear-gradient(to bottom, 
                rgba(255,255,255,0.95) 0%,
                rgba(255,255,255,0.85) 50%,
                rgba(255,255,255,0.95) 100%),
                url("https://images.unsplash.com/photo-1488646953014-85cb44e25828");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        
        /* Container styling */
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        /* Title styling */
        .title-text {
            background: linear-gradient(135deg, #2193b0, #6dd5ed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            text-align: center;
            padding: 2rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Subtitle styling */
        .subtitle-text {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 3rem;
            line-height: 1.6;
        }
        
        /* Card styling */
        .attraction-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 25px 0;
            transition: all 0.3s ease;
            border-left: 5px solid #2193b0;
        }
        
        .attraction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #2193b0, #6dd5ed);
            color: white;
            border: none;
            padding: 1rem 2.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            width: 100%;
            max-width: 300px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(33,147,176,0.3);
        }
        
        /* Input field styling */
        .stTextArea > div > div {
            border-radius: 15px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
            background: white;
        }
        
        .stTextArea > div > div:focus-within {
            border-color: #2193b0;
            box-shadow: 0 0 0 3px rgba(33,147,176,0.2);
        }
        
        /* Warning and error styling */
        .stAlert {
            border-radius: 10px;
            border: none;
            padding: 1rem;
        }
        
        /* Loading animation */
        .stSpinner {
            text-align: center;
            padding: 2rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .title-text {
                font-size: 2.5rem !important;
            }
            .content-container {
                padding: 1rem;
            }
            .attraction-card {
                padding: 15px;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def load_resources():
    """Load pipeline and dataset with error handling."""
    try:
        with st.spinner("Loading travel resources..."):
            pipeline = pickle.load(open("destination_pipeline.pkl", "rb"))
            data = pd.read_csv("best_travel_destinations_for_2025.csv")
        return pipeline, data
    except FileNotFoundError as e:
        st.error("Required files not found. Please check if all resources are present.")
        return None, None
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="AI Travel Advisor",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    set_custom_style()
    
    with st.container():
        st.markdown('<h1 class="title-text">✈️ AI Travel Advisor</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="subtitle-text">
            Let our AI help you discover your perfect destination! Share your dream activities, 
            and we'll suggest the ideal places tailored just for you. Try mentioning activities 
            like "hiking mountain trails", "exploring ancient ruins", or "relaxing on beaches".
        </div>
        """, unsafe_allow_html=True)
        
        pipeline, data = load_resources()
        if not pipeline or data is None:
            st.stop()
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            user_query = st.text_area(
                "What's your ideal travel experience?",
                height=120,
                placeholder="E.g., I want to go hiking in mountains, explore local markets, and try authentic street food..."
            )
            
            recommend_button = st.button("🔍 Find My Perfect Destination", use_container_width=True)
        
        if recommend_button:
            if not user_query.strip():
                st.warning("Please tell us about your desired travel experience!")
                st.stop()
                
            try:
                with st.spinner("🌍 Exploring the perfect destinations for you..."):
                    result = recommend_attractions(user_query, pipeline, data)
                    
                if isinstance(result, str):
                    st.warning(result)
                else:
                    predicted_country, recommendations = result
                    
                    st.markdown(f"""
                        <div style='text-align: center; padding: 3rem;'>
                            <h2 style='color: #2193b0; margin-bottom: 1rem;'>Based on your interests, we recommend:</h2>
                            <h1 style='color: #2193b0; font-size: 3rem; margin-bottom: 2rem;'>{predicted_country} 🎯</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                            <div class='attraction-card'>
                                <h3 style='color: #2193b0; margin-bottom: 1rem;'>
                                    {row['Attraction']} ✨
                                </h3>
                                <p style='color: #666; margin-bottom: 0.5rem;'>
                                    <strong>Location:</strong> {row['Country']}
                                </p>
                                <p style='color: #444; line-height: 1.6;'>
                                    {row['Description']}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error("Oops! Something went wrong while processing your request. Please try again.")
                print(f"Error: {str(e)}")  # For debugging

if __name__ == "__main__":
    main()