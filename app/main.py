import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configurations ---
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

# --- Load Models & Data ---
@st.cache_resource
def load_models_and_data():
    models_dir = "data/processed/models"
    
    # Load Models
    reg_model = joblib.load(os.path.join(models_dir, 'regression_model.pkl'))
    clf_model = joblib.load(os.path.join(models_dir, 'classification_model.pkl'))
    
    # Load Assets
    user_item = pd.read_csv(os.path.join(models_dir, 'user_item_matrix.csv'))
    attractions = pd.read_csv(os.path.join(models_dir, 'attractions.csv'))
    
    # Modes mapping
    # 1: Business, 2: Couples, 3: Family, 4: Friends, 5: Solo (Assumed based on typical modes)
    mode_mapping = {1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"}
    
    return reg_model, clf_model, user_item, attractions, mode_mapping

reg_model, clf_model, user_item_matrix, attractions, mode_mapping = load_models_and_data()

# --- Main App ---
st.title("🌍 Tourism Experience Analytics")
st.markdown("### Classification, Prediction, and Recommendation System")

tab1, tab2, tab3 = st.tabs(["Prediction & Classification", "Personalized Recommendations", "Tourism Analytics (EDA)"])

with tab1:
    st.header("Predict Visit Mode & Rating")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Input")
        user_id = st.number_input("User ID", value=14, step=1)
        attraction_id = st.number_input("Attraction ID", value=640, step=1)
        visit_year = st.number_input("Visit Year", value=2024, step=1)
        visit_month = st.selectbox("Visit Month", list(range(1, 13)))
        
        continent_id = st.number_input("Continent ID", value=5, step=1)
        region_id = st.number_input("Region ID", value=20, step=1)
        country_id = st.number_input("Country ID", value=155, step=1)
        city_id = st.number_input("City ID", value=220, step=1)
        attraction_type_id = st.number_input("Attraction Type ID", value=2, step=1)

        with st.expander("ℹ️ Why do we need this data? (Consent to Submit)"):
            st.write(
                "**What we do with your data:** We analyze your demographic information (like your region or country) "
                "and your travel context (month, year) alongside historical data.\n\n"
                "**Why you should provide it:** Sharing this information allows our Machine Learning models to provide highly "
                "**personalized attraction recommendations** and accurately predict if an attraction matches your preferences. "
                "This ensures you have the best possible tourism experience without sifting through irrelevant options. "
                "Your data is solely used to enhance your travel recommendations."
            )

    with col2:
        st.subheader("Results")
        if st.button("Submit & Predict"):
            # Prepare feature array: ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 
            # 'ContinentId_x', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId']
            features = np.array([[user_id, attraction_id, visit_year, visit_month,
                                 continent_id, region_id, country_id, city_id, attraction_type_id]])
            
            # 1. Classification
            pred_mode_id = clf_model.predict(features)[0]
            pred_mode = mode_mapping.get(pred_mode_id, f"Mode {pred_mode_id}")
            
            # 2. Regression
            pred_rating = reg_model.predict(features)[0]
            
            st.success(f"**Predicted Visit Mode:** {pred_mode}")
            st.info(f"**Predicted Rating:** {pred_rating:.2f} / 5.00")
            
            # Show Attraction Info if available
            att_info = attractions[attractions['AttractionId'] == attraction_id]
            if not att_info.empty:
                st.write("**Attraction Name:**", att_info.iloc[0]['Attraction'])
                st.write("**Type:**", att_info.iloc[0]['AttractionType'])
                st.write("**Location:**", att_info.iloc[0]['AttractionAddress'])

with tab2:
    st.header("Top Recommended Attractions")
    rec_user_id = st.number_input("Enter User ID for Recommendations", value=14, step=1)
    
    with st.expander("ℹ️ Why do we need your User ID? (Consent to Submit)"):
        st.write(
            "**What we do:** We look up your past travel history and ratings. We then compare your preferences with similar travelers.\n\n"
            "**Why provide it:** By knowing what you've enjoyed in the past, our system filters out places you've already visited "
            "and highlights hidden gems and top-rated attractions tailored specifically to your taste."
        )
    
    if st.button("Submit & Get Recommendations"):
        # Simple Collaborative Filtering (Item-Based naive / User Top Rated)
        # We find top rated attractions for similar users, or just popular ones.
        # For simplicity, returning the top rated overall, or top rated by user history.
        
        user_history = user_item_matrix[user_item_matrix['UserId'] == rec_user_id]
        if not user_history.empty:
            st.write(f"Found history for User {rec_user_id}. Showing top recommendations based on high overall ratings:")
            # Find highly rated overall attractions not visited by user
            visited = user_history['AttractionId'].tolist()
            overall_ratings = user_item_matrix.groupby('AttractionId')['Rating'].mean().reset_index()
            recommendations = overall_ratings[~overall_ratings['AttractionId'].isin(visited)].sort_values(by='Rating', ascending=False).head(5)
            
            rec_full = recommendations.merge(attractions, on='AttractionId', how='left')
            st.table(rec_full[['Attraction', 'AttractionType', 'Rating', 'AttractionAddress']])
        else:
            st.warning(f"No history found for User {rec_user_id}. Showing popular attractions:")
            popular = user_item_matrix.groupby('AttractionId').agg({'Rating': 'mean', 'UserId': 'count'}).reset_index()
            # filter to at least a few reviews
            popular = popular[popular['UserId'] > 5].sort_values(by='Rating', ascending=False).head(5)
            rec_full = popular.merge(attractions, on='AttractionId', how='left')
            st.table(rec_full[['Attraction', 'AttractionType', 'Rating', 'AttractionAddress']])

with tab3:
    st.header("Exploratory Data Analysis")
    st.write("Visualizations generated during the data preparation phase.")
    
    # Load and display images
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image("notebooks/user_distribution_continent.png", caption="User Distribution Across Continents", use_container_width=True)
            st.image("notebooks/attraction_types_popularity.png", caption="Attraction Types Popularity", use_container_width=True)
        with col2:
            st.image("notebooks/rating_by_visit_mode.png", caption="Rating by Visit Mode", use_container_width=True)
    except FileNotFoundError:
        st.error("EDA images not found. Please ensure the EDA script ran successfully.")
