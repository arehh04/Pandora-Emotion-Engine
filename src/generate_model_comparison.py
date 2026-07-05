import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_comparison_chart():
    # Hardcoded results from our evaluation table
    models = ['Ridge (Baseline)', 'XGBoost (Tuned)', 'Random Forest (Tuned)']
    rmse_scores = [29.18, 28.29, 28.22]
    r2_scores = [0.103, 0.158, 0.162]
    
    # Set up the matplotlib figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # X axis positions
    x = np.arange(len(models))
    width = 0.35
    
    # Plot RMSE on primary y-axis
    rects1 = ax1.bar(x - width/2, rmse_scores, width, label='RMSE (Lower is Better)', color='#e74c3c')
    ax1.set_ylabel('Root Mean Square Error (RMSE)', color='#c0392b', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    ax1.set_ylim(27, 30) # Zoomed in to show the difference clearly
    
    # Create secondary y-axis for R^2
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, r2_scores, width, label='R² Score (Higher is Better)', color='#3498db')
    ax2.set_ylabel('R² Score', color='#2980b9', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2980b9')
    ax2.set_ylim(0, 0.2)
    
    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12, fontweight='bold')
    
    # Title
    plt.title('Model Performance Comparison (Baseline vs Feature Fusion)', fontsize=14, fontweight='bold')
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Add value labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#c0392b')
                    
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2980b9')

    plt.tight_layout()
    
    # Save image
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'models', 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison chart generated at: {output_path}")

if __name__ == "__main__":
    generate_comparison_chart()
