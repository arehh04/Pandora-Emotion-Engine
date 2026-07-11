import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    thesis_dir = os.path.join(base_dir, "thesis_output")
    os.makedirs(thesis_dir, exist_ok=True)
    
    # 1. Load Original Data
    print("Loading Original Data...")
    df_clean = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    df_tokens = pd.read_csv(os.path.join(data_dir, "train_tokens.csv"))
    df_orig = df_clean.copy()
    df_orig['extraversion'] = df_tokens['extraversion']
    
    # 2. Load Augmented Data
    print("Loading Augmented Data...")
    df_aug = pd.read_csv(os.path.join(data_dir, "train_augmented.csv"))
    
    # 3. Bin Both
    bins = [0, 33, 66, 100]
    labels = ['Low', 'Medium', 'High']
    
    df_orig['Bin'] = pd.cut(df_orig['extraversion'], bins=bins, labels=labels, include_lowest=True)
    df_aug['Bin'] = pd.cut(df_aug['extraversion'], bins=bins, labels=labels, include_lowest=True)
    
    counts_orig = df_orig['Bin'].value_counts().reindex(labels)
    counts_aug = df_aug['Bin'].value_counts().reindex(labels)
    
    # 4. Prepare for plotting
    plot_df = pd.DataFrame({
        'Bin': labels * 2,
        'Count': list(counts_orig.values) + list(counts_aug.values),
        'Dataset': ['Before Augmentation'] * 3 + ['After LLM Augmentation (Gemma 3)'] * 3
    })
    
    # 5. Plot Side-by-Side Bar Chart
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=plot_df, x='Bin', y='Count', hue='Dataset', palette=['#b0929c', '#ff2a6d'])
    
    plt.title('Distribution of Extraversion (Social Expressiveness) Before and After LLM Augmentation', fontsize=14, fontweight='bold')
    plt.xlabel('Expressiveness Bin', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.legend(title='Dataset')
    
    # Add exact numbers on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=10, color='black', fontweight='bold')
                   
    # Save the plot
    out_path = os.path.join(thesis_dir, "augmentation_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to {out_path}")

if __name__ == "__main__":
    main()
