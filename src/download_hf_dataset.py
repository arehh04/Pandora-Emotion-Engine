import os
import pandas as pd
from datasets import load_dataset

def main():
    print("Loading Pandora Personality Dataset...")
    ds = load_dataset("marchmallow/Automated-Personality-Prediction")
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save train, validation, and test/eval sets
    for split in ds.keys():
        print(f"Saving {split} set to CSV...")
        df = ds[split].to_pandas()
        csv_path = os.path.join(data_dir, f"{split}_set.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {split}_set.csv with {len(df)} records.")

if __name__ == "__main__":
    main()
