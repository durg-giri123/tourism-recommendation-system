import pandas as pd
import os
import argparse

def load_data(data_dir):
    print("Loading datasets...")
    # Load all datasets
    city_df = pd.read_excel(os.path.join(data_dir, "City.xlsx"))
    continent_df = pd.read_excel(os.path.join(data_dir, "Continent.xlsx"))
    country_df = pd.read_excel(os.path.join(data_dir, "Country.xlsx"))
    mode_df = pd.read_excel(os.path.join(data_dir, "Mode.xlsx"))
    region_df = pd.read_excel(os.path.join(data_dir, "Region.xlsx"))
    transaction_df = pd.read_excel(os.path.join(data_dir, "Transaction.xlsx"))
    type_df = pd.read_excel(os.path.join(data_dir, "Type.xlsx"))
    
    # Try using Updated_Item if it exists
    item_file = "Updated_Item.xlsx"
    if not os.path.exists(os.path.join(data_dir, item_file)):
        item_file = "Item.xlsx"
    item_df = pd.read_excel(os.path.join(data_dir, item_file))
    user_df = pd.read_excel(os.path.join(data_dir, "User.xlsx"))
    
    return {
        "City": city_df,
        "Continent": continent_df,
        "Country": country_df,
        "Mode": mode_df,
        "Region": region_df,
        "Transaction": transaction_df,
        "Type": type_df,
        "Item": item_df,
        "User": user_df
    }

def merge_data(dfs):
    print("Merging datasets...")
    
    # Start with Transactions
    df = dfs["Transaction"].copy()
    
    # Merge User details
    df = df.merge(dfs["User"], on="UserId", how="left")
    
    # Merge Continent, Region, Country, City (User Demographics)
    df = df.merge(dfs["Continent"], on="ContinentId", how="left")
    df = df.merge(dfs["Region"], on="RegionId", how="left")
    df = df.merge(dfs["Country"], on="CountryId", how="left", suffixes=("", "_Country"))
    df = df.merge(dfs["City"], on="CityId", how="left", suffixes=("", "_City"))
        
    # Merge Item details (AttractionCityId -> CityName etc, AttractionTypeId -> AttractionType)
    # The attraction id is 'AttractionId'
    item_df = dfs["Item"].copy()
    df = df.merge(item_df, on="AttractionId", how="left")
    
    # Merge Attraction Type
    df = df.merge(dfs["Type"], on="AttractionTypeId", how="left")
        
    # Merge Visit Mode
    # Transaction has 'VisitMode' with values like 2, 4. Mode has 'VisitModeId' and 'VisitMode' (string).
    # We should rename Transaction's VisitMode to VisitModeId
    df = df.rename(columns={"VisitMode": "VisitModeId"})
    mode_df = dfs["Mode"].rename(columns={"VisitMode": "VisitModeString"})
    df = df.merge(mode_df, on="VisitModeId", how="left")
    df = df.rename(columns={"VisitModeString": "VisitMode"})
        
    return df

def clean_and_encode(df):
    print("Cleaning data...")
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # Drop records with missing target (Rating or VisitMode)
    if 'Rating' in df.columns:
        df = df.dropna(subset=['Rating'])
        
    # Standardize data types, handle NaN
    # Standardize data types, handle NaN
    for col in ['CityId', 'CountryId', 'RegionId', 'ContinentId', 'AttractionTypeId']:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)
        else:
            print(f"Warning: {col} not found in columns: {df.columns}")


    
    print("Feature Engineering...")
    # Add any basic feature engineering here if needed
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"data")
    parser.add_argument("--output_dir", type=str, default=r"data\processed")
    args = parser.parse_args()
    
    data_dir_full = os.path.abspath(args.data_dir)
    output_dir_full = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_full, exist_ok=True)
    
    dfs = load_data(data_dir_full)
    merged_df = merge_data(dfs)
    cleaned_df = clean_and_encode(merged_df)
    
    output_path = os.path.join(output_dir_full, "merged_data.csv")
    print(f"Saving merged data to {output_path}...")
    cleaned_df.to_csv(output_path, index=False)
    print("Data processing complete. Shape:", cleaned_df.shape)
