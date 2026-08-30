import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. User distribution across continents
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Continent', data=df, order=df['Continent'].value_counts().index)
    plt.title('User Distribution Across Continents')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_distribution_continent.png'))
    plt.close()
    
    # 2. Attraction Types Popularity
    plt.figure(figsize=(12, 8))
    sns.countplot(y='AttractionType', data=df, order=df['AttractionType'].value_counts().index[:15])
    plt.title('Top 15 Most Visited Attraction Types')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attraction_types_popularity.png'))
    plt.close()
    
    # 3. VisitMode and Ratings
    if 'VisitMode' in df.columns and 'Rating' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='VisitMode', y='Rating', data=df)
        plt.title('Rating Distribution by Visit Mode')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_by_visit_mode.png'))
        plt.close()

if __name__ == "__main__":
    df = pd.read_csv(r"data\processed\merged_data.csv")
    perform_eda(df, "notebooks")
    print("EDA charts saved to notebooks/ folder.")
