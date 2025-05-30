import pandas as pd
import os

# --- Configuration ---
BASE_DIR = './data' # This will be bind-mounted as /data in Docker
RAW_RECOMMENDATIONS_FILE = 'realrecommendations.csv' # Your original recommendations file
RAW_GAMES_FILE = 'games.csv'
RAW_USERS_FILE = 'users.csv'

SAMPLED_RECOMMENDATIONS_FILE = os.path.join(BASE_DIR, 'recs_5m.csv')
SAMPLED_GAMES_FILE = os.path.join(BASE_DIR, 'games_sampled.csv')
SAMPLED_USERS_FILE = os.path.join(BASE_DIR, 'users_sampled.csv')

NUM_REC_ROWS = 5_000_000

# --- Ensure ./data directory exists ---
os.makedirs(BASE_DIR, exist_ok=True)

# --- 1. Process recommendations.csv ---
print(f"Processing {RAW_RECOMMENDATIONS_FILE}...")
try:
    df_recs = pd.read_csv(RAW_RECOMMENDATIONS_FILE, usecols=['user_id', 'app_id', 'is_recommended', 'hours', 'date'])
    
    if len(df_recs) > NUM_REC_ROWS:
        print(f"Sampling {NUM_REC_ROWS} rows from recommendations...")
        df_recs = df_recs.sample(n=NUM_REC_ROWS, random_state=42)
    else:
        print(f"Using all {len(df_recs)} rows from recommendations as it's less than or equal to {NUM_REC_ROWS}.")

    # Rename columns to match plan.md
    df_recs.rename(columns={'is_recommended': 'recommended', 'date': 'timestamp'}, inplace=True)
    
    # Convert 'recommended' to int (0 or 1)
    df_recs['recommended'] = df_recs['recommended'].astype(int)
    
    # Ensure 'hours' is numeric, fill NaNs if any (e.g., with 0)
    df_recs['hours'] = pd.to_numeric(df_recs['hours'], errors='coerce').fillna(0)

    # Select final columns
    df_recs = df_recs[['user_id', 'app_id', 'recommended', 'hours', 'timestamp']]

    df_recs.to_csv(SAMPLED_RECOMMENDATIONS_FILE, index=False)
    print(f"Saved {len(df_recs)} rows to {SAMPLED_RECOMMENDATIONS_FILE}")

    # Get unique IDs for sub-sampling other files
    unique_app_ids = df_recs['app_id'].unique()
    unique_user_ids = df_recs['user_id'].unique()

    # --- 2. Process games.csv ---
    print(f"\nProcessing {RAW_GAMES_FILE}...")
    if os.path.exists(RAW_GAMES_FILE):
        df_games = pd.read_csv(RAW_GAMES_FILE)
        # Filter games that appear in the sampled recommendations
        df_games_sampled = df_games[df_games['app_id'].isin(unique_app_ids)]
        df_games_sampled = df_games_sampled[['app_id', 'title', 'price_final']]
        df_games_sampled.to_csv(SAMPLED_GAMES_FILE, index=False)
        print(f"Saved {len(df_games_sampled)} games to {SAMPLED_GAMES_FILE} (based on sampled recommendations)")
    else:
        print(f"Warning: {RAW_GAMES_FILE} not found. Skipping games sub-sampling.")
        # Create an empty file with headers if models depend on it, to avoid read errors
        pd.DataFrame(columns=['app_id', 'title', 'price_final']).to_csv(SAMPLED_GAMES_FILE, index=False)


    # --- 3. Process users.csv ---
    print(f"\nProcessing {RAW_USERS_FILE}...")
    if os.path.exists(RAW_USERS_FILE):
        df_users = pd.read_csv(RAW_USERS_FILE)
        # Filter users that appear in the sampled recommendations
        df_users_sampled = df_users[df_users['user_id'].isin(unique_user_ids)]
        # Select minimal useful columns
        df_users_sampled = df_users_sampled[['user_id', 'products', 'reviews']] # Or just user_id if others aren't used
        df_users_sampled.to_csv(SAMPLED_USERS_FILE, index=False)
        print(f"Saved {len(df_users_sampled)} users to {SAMPLED_USERS_FILE} (based on sampled recommendations)")
    else:
        print(f"Warning: {RAW_USERS_FILE} not found. Skipping users sub-sampling.")
        pd.DataFrame(columns=['user_id', 'products', 'reviews']).to_csv(SAMPLED_USERS_FILE, index=False)

    print("\nPreprocessing complete.")
    print(f"Ensure these files are in the './data/' directory before running 'docker compose up':")
    print(f"- {os.path.basename(SAMPLED_RECOMMENDATIONS_FILE)}")
    print(f"- {os.path.basename(SAMPLED_GAMES_FILE)}")
    print(f"- {os.path.basename(SAMPLED_USERS_FILE)}")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the raw CSV files ({RAW_RECOMMENDATIONS_FILE}, {RAW_GAMES_FILE}, {RAW_USERS_FILE}) are in the same directory as this script or provide correct paths.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")