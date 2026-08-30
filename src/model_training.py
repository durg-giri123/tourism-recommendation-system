import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    # Features for models
    features = ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 
                'ContinentId_x', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId']
    # Use only available columns
    features = [f for f in features if f in df.columns]
    
    X = df[features].fillna(-1)
    return X

def train_regression(df, models_dir):
    print("Training Regression Model (Predicting Rating)...")
    X = prepare_features(df)
    
    # Target
    y = df['Rating']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Regression MSE: {mse:.4f}")
    
    joblib.dump(model, os.path.join(models_dir, 'regression_model.pkl'))
    return model

def train_classification(df, models_dir):
    print("Training Classification Model (Predicting VisitMode)...")
    X = prepare_features(df)
    
    # Target
    y = df['VisitModeId']
    
    # We might have some NaNs in VisitModeId if not found, drop them
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Classification Accuracy: {acc:.4f}")
    
    joblib.dump(model, os.path.join(models_dir, 'classification_model.pkl'))
    return model

def build_recommendation(df, models_dir):
    print("Building Recommendation Matrix...")
    # Simple User-Item Matrix
    user_item = df.groupby(['UserId', 'AttractionId'])['Rating'].mean().reset_index()
    user_item.to_csv(os.path.join(models_dir, 'user_item_matrix.csv'), index=False)
    
    # Attraction details for fast lookup
    attractions = df[['AttractionId', 'Attraction', 'AttractionType', 'AttractionAddress']].drop_duplicates()
    attractions.to_csv(os.path.join(models_dir, 'attractions.csv'), index=False)
    print("Saved recommendation assets.")

if __name__ == "__main__":
    df = pd.read_csv(r"data\processed\merged_data.csv")
    models_dir = r"data\processed\models"
    os.makedirs(models_dir, exist_ok=True)
    
    train_regression(df, models_dir)
    train_classification(df, models_dir)
    build_recommendation(df, models_dir)
