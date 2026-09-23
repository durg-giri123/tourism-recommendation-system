import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def prepare_features(df):
    # Features for models
    features = ['UserId', 'AttractionId', 'VisitYear', 'VisitMonth', 
                'ContinentId_x', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId']
    # Use only available columns
    features = [f for f in features if f in df.columns]
    
    X = df[features].fillna(-1)
    return X, features

def train_regression(df, models_dir):
    print("--- Training Regression Model (Predicting Rating) ---")
    X, feature_names = prepare_features(df)
    y = df['Rating']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    print(f"5-Fold CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full train set for final model and metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Test Set Metrics -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    
    # Save Model
    joblib.dump(model, os.path.join(models_dir, 'regression_model.pkl'))
    
    # Save Feature Importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importances.to_csv(os.path.join(models_dir, 'regression_feature_importances.csv'), index=False)
    
    return model

def train_classification(df, models_dir):
    print("\n--- Training Classification Model (Predicting VisitMode) ---")
    X, feature_names = prepare_features(df)
    y = df['VisitModeId']
    
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # Using 'weighted' average since it's a multi-class problem
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    print(f"Test Set Metrics -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    joblib.dump(model, os.path.join(models_dir, 'classification_model.pkl'))
    
    # Save Feature Importances
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importances.to_csv(os.path.join(models_dir, 'classification_feature_importances.csv'), index=False)
    
    return model

def build_recommendation(df, models_dir):
    print("\n--- Building Recommendation Matrix ---")
    user_item = df.groupby(['UserId', 'AttractionId'])['Rating'].mean().reset_index()
    user_item.to_csv(os.path.join(models_dir, 'user_item_matrix.csv'), index=False)
    
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
