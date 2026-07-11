import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    print("Loading original clean data...")
    df_clean = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    df_tokens = pd.read_csv(os.path.join(data_dir, "train_tokens.csv"))
    
    # Merge properly
    df_orig = df_clean.copy()
    df_orig['extraversion'] = df_tokens['extraversion']
    print(f"Original clean rows: {len(df_orig)}")
    
    print("Loading corrupted augmented data...")
    df_aug = pd.read_csv(os.path.join(data_dir, "train_augmented.csv"))
    
    # Extract only the augmented rows which are perfectly fine
    valid_aug = df_aug[df_aug['type'] == '_AUGMENTED'].copy()
    print(f"Valid augmented rows recovered: {len(valid_aug)}")
    
    # Check if 'type' exists in original, if not add it
    if 'type' not in df_orig.columns:
        df_orig['type'] = 'ORIGINAL'
        
    # Concatenate
    final_df = pd.concat([df_orig, valid_aug], ignore_index=True)
    print(f"Final perfect dataset size: {len(final_df)}")
    
    # Verify no NaNs in extraversion
    nans = final_df['extraversion'].isna().sum()
    print(f"NaNs in extraversion: {nans}")
    
    if nans == 0:
        out_path = os.path.join(data_dir, "train_augmented.csv")
        final_df.to_csv(out_path, index=False)
        print("Successfully rescued and saved pristine train_augmented.csv!")
    else:
        print("ERROR: Still NaNs present.")

if __name__ == "__main__":
    main()
