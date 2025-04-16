import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib.colors import LinearSegmentedColormap

"""
Generated with Claude
"""

def load_data(file_path):
    """Load and validate the individual analysis CSV file."""
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Check that required columns exist
    required_columns = ['id', 'similarity', 'interviewer_length', 'interviewee_length', 'ratio']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check if cluster column exists
    has_clusters = 'cluster' in df.columns
    
    print(f"Loaded data for {len(df)} transcripts.")
    if has_clusters:
        print(f"Found {df['cluster'].nunique()} clusters.")
        
    return df, has_clusters

def create_visualization_dir(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def create_similarity_histogram(df, output_dir):
    """Create histogram of similarity scores."""
    plt.figure(figsize=(10, 6))
    
    # Create custom colormap from red to green
    cmap = LinearSegmentedColormap.from_list('red_to_green', ['#f73b3b', '#ffae49', '#66b266'])
    
    # Create bins and normalize colors
    bins = np.linspace(df['similarity'].min(), df['similarity'].max(), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    norm = plt.Normalize(min(bin_centers), max(bin_centers))
    
    # Plot histogram with color gradient
    n, bins, patches = plt.hist(df['similarity'], bins=bins, alpha=0.7, edgecolor='black')
    
    # Set colors for each bin based on value
    for i, patch in enumerate(patches):
        color = cmap(norm(bin_centers[i]))
        patch.set_facecolor(color)
    
    plt.title('Distribution of Interviewer-Interviewee Similarity Scores', fontsize=14)
    plt.xlabel('Similarity Score (closer to 1 = more similar)', fontsize=12)
    plt.ylabel('Number of Interviews', fontsize=12)
    
    # Add vertical lines for interpretation
    plt.axvline(x=0.7, color='#ff7700', linestyle='--', alpha=0.7, 
                label='0.7: Moderate similarity')
    plt.axvline(x=0.85, color='#00aa00', linestyle='--', alpha=0.7, 
                label='0.85: High similarity')
    
    # Add mean line
    plt.axvline(x=df['similarity'].mean(), color='black', linestyle='-', alpha=0.7,
                label=f'Mean: {df["similarity"].mean():.3f}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = os.path.join(output_dir, 'similarity_histogram.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity histogram to {output_file}")

def create_length_comparison(df, output_dir):
    """Create scatter plot comparing interviewer and interviewee text lengths."""
    plt.figure(figsize=(12, 8))
    
    # Calculate point sizes based on similarity (higher similarity = larger point)
    sizes = 20 + (df['similarity'] * 100)
    
    # Create scatter plot
    scatter = plt.scatter(df['interviewer_length'], df['interviewee_length'], 
                         s=sizes, alpha=0.6, c=df['similarity'], cmap='viridis',
                         edgecolors='black', linewidths=0.5)
    
    # Add diagonal line representing equal length
    max_val = max(df['interviewer_length'].max(), df['interviewee_length'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.4, label='Equal Length')
    
    # Add ratio lines
    for ratio in [2, 3, 4]:
        plt.plot([0, max_val], [0, max_val * ratio], 'k:', alpha=0.2, 
                label=f'Ratio {ratio}:1')
    
    plt.title('Interviewer vs. Interviewee Text Length', fontsize=14)
    plt.xlabel('Interviewer Text Length (characters)', fontsize=12)
    plt.ylabel('Interviewee Text Length (characters)', fontsize=12)
    
    plt.colorbar(scatter, label='Similarity Score')
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left')
    plt.xlim(0,3000)
    plt.ylim(0,10000)
    
    output_file = os.path.join(output_dir, 'length_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved length comparison plot to {output_file}")

def create_ratio_similarity_plot(df, output_dir):
    """Create scatter plot showing relationship between ratio and similarity."""
    plt.figure(figsize=(12, 6))
    
    # Cap extreme ratios for better visualization
    capped_ratio = df['ratio'].copy()
    cap_value = np.percentile(df['ratio'], 95)  # Cap at 95th percentile
    capped_ratio[capped_ratio > cap_value] = cap_value
    
    # Create scatter plot
    scatter = plt.scatter(capped_ratio, df['similarity'], s=80, alpha=0.7, 
                         c=df['interviewee_length'], cmap='plasma',
                         edgecolors='black', linewidths=0.5)
    
    # Add trend line
    z = np.polyfit(capped_ratio, df['similarity'], 1)
    p = np.poly1d(z)
    plt.plot(np.sort(capped_ratio), p(np.sort(capped_ratio)), "r--", alpha=0.7,
            label=f'Trend line (slope: {z[0]:.4f})')
    
    plt.title('Interviewee-to-Interviewer Ratio vs. Similarity Score', fontsize=14)
    plt.xlabel('Length Ratio (Interviewee/Interviewer)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    
    plt.colorbar(scatter, label='Interviewee Text Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = os.path.join(output_dir, 'ratio_similarity.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ratio-similarity plot to {output_file}")

def create_cluster_visualization(df, output_dir):
    """Create visualizations related to clusters if cluster data is present."""
    if 'cluster' not in df.columns:
        print("No cluster data found. Skipping cluster visualizations.")
        return
    
    # 1. Cluster size bar chart
    plt.figure(figsize=(10, 6))
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax = cluster_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title('Interview Clusters by Size', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Interviews', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels to bars
    for i, count in enumerate(cluster_counts):
        ax.text(i, count + 0.1, str(count), ha='center', fontsize=10)
    
    output_file = os.path.join(output_dir, 'cluster_sizes.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster size chart to {output_file}")
    
    # 2. Similarity by cluster boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='similarity', data=df, palette='viridis')
    
    plt.title('Similarity Scores by Cluster', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    output_file = os.path.join(output_dir, 'cluster_similarities.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster similarities chart to {output_file}")
    
    # 3. Ratio by cluster boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='ratio', data=df, palette='plasma')
    
    plt.title('Interviewee/Interviewer Ratio by Cluster', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Length Ratio', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Cap y-axis to reasonable values
    upper_limit = np.percentile(df['ratio'], 95)
    plt.ylim(0, upper_limit * 1.1)
    
    output_file = os.path.join(output_dir, 'cluster_ratios.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster ratios chart to {output_file}")
    
    # 4. Cluster characteristics heatmap
    plt.figure(figsize=(14, 8))
    
    # Prepare data for heatmap
    cluster_metrics = df.groupby('cluster').agg({
        'similarity': 'mean',
        'interviewer_length': 'mean',
        'interviewee_length': 'mean',
        'ratio': 'mean',
        'id': 'count'  # Count for size
    }).rename(columns={'id': 'count'})
    
    # Normalize values for better visualization
    normalized = cluster_metrics.copy()
    for col in normalized.columns:
        normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
    
    # Create heatmap
    sns.heatmap(normalized, annot=cluster_metrics.round(2), fmt='.2f', cmap='YlGnBu',
               linewidths=0.5, cbar_kws={'label': 'Normalized Value'})
    
    plt.title('Cluster Characteristics Comparison', fontsize=14)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'cluster_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster characteristics heatmap to {output_file}")

def create_top_bottom_analysis(df, output_dir):
    """Create visualizations comparing top and bottom interviews by similarity."""
    # Sort by similarity
    sorted_df = df.sort_values('similarity')
    
    # Get top and bottom 10% (or at least 5 interviews)
    n_samples = max(int(len(df) * 0.1), 5)
    bottom_df = sorted_df.head(n_samples)
    top_df = sorted_df.tail(n_samples)
    
    # Create comparative bar chart
    plt.figure(figsize=(12, 8))
    
    # Set up bars
    labels = ['Similarity', 'Interviewer Length (÷100)', 
              'Interviewee Length (÷100)', 'Length Ratio']
    
    bottom_means = [
        bottom_df['similarity'].mean(),
        bottom_df['interviewer_length'].mean() / 100,
        bottom_df['interviewee_length'].mean() / 100,
        bottom_df['ratio'].mean()
    ]
    
    top_means = [
        top_df['similarity'].mean(),
        top_df['interviewer_length'].mean() / 100,
        top_df['interviewee_length'].mean() / 100,
        top_df['ratio'].mean()
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, bottom_means, width, label='Bottom Similarity Interviews', 
                   color='#ff7766', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, top_means, width, label='Top Similarity Interviews', 
                   color='#66aa77', edgecolor='black', linewidth=1)
    
    # Add labels and formatting
    ax.set_title('Comparison of Top vs. Bottom Interviews by Similarity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    add_labels(rects1)
    add_labels(rects2)
    
    output_file = os.path.join(output_dir, 'top_bottom_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top-bottom comparison to {output_file}")
    
    # Create top/bottom IDs list for reference
    with open(os.path.join(output_dir, 'top_bottom_ids.txt'), 'w') as f:
        f.write("Top Interview IDs by Similarity:\n")
        for idx, row in top_df.iterrows():
            f.write(f"  ID: {row['id']}, Similarity: {row['similarity']:.4f}\n")
        
        f.write("\nBottom Interview IDs by Similarity:\n")
        for idx, row in bottom_df.iterrows():
            f.write(f"  ID: {row['id']}, Similarity: {row['similarity']:.4f}\n")
    
    print(f"Saved top and bottom interview IDs to {os.path.join(output_dir, 'top_bottom_ids.txt')}")

def create_summary_report(df, output_dir, has_clusters):
    """Create a summary report of the visualizations."""
    report_path = os.path.join(output_dir, 'visualization_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("INTERVIEW ANALYSIS VISUALIZATION SUMMARY\n")
        f.write("=======================================\n\n")
        
        f.write("Dataset Statistics:\n")
        f.write(f"  Number of Interviews: {len(df)}\n")
        f.write(f"  Average Similarity: {df['similarity'].mean():.4f}\n")
        f.write(f"  Similarity Range: {df['similarity'].min():.4f} to {df['similarity'].max():.4f}\n")
        f.write(f"  Average Interviewer Length: {df['interviewer_length'].mean():.1f} characters\n")
        f.write(f"  Average Interviewee Length: {df['interviewee_length'].mean():.1f} characters\n")
        f.write(f"  Average Length Ratio: {df['ratio'].mean():.2f}\n\n")
        
        if has_clusters:
            f.write(f"Cluster Information:\n")
            cluster_counts = df['cluster'].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                cluster_df = df[df['cluster'] == cluster]
                f.write(f"  Cluster {cluster}:\n")
                f.write(f"    Size: {count} interviews\n")
                f.write(f"    Average Similarity: {cluster_df['similarity'].mean():.4f}\n")
                f.write(f"    Average Ratio: {cluster_df['ratio'].mean():.2f}\n")
            f.write("\n")
        
        f.write("Visualization Files:\n")
        f.write("  similarity_histogram.png - Distribution of similarity scores\n")
        f.write("  length_comparison.png - Interviewer vs. interviewee text lengths\n")
        f.write("  ratio_similarity.png - Relationship between length ratio and similarity\n")
        f.write("  top_bottom_comparison.png - Comparison of highest and lowest similarity interviews\n")
        
        if has_clusters:
            f.write("  cluster_sizes.png - Number of interviews in each cluster\n")
            f.write("  cluster_similarities.png - Distribution of similarity scores by cluster\n")
            f.write("  cluster_ratios.png - Distribution of length ratios by cluster\n")
            f.write("  cluster_heatmap.png - Heat map showing key metrics for each cluster\n")
    
    print(f"Created summary report at {report_path}")

def visualize_interviews(input_file, output_dir='visualizations'):
    """Main function to create all visualizations."""
    # Set Seaborn style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Load data
    df, has_clusters = load_data(input_file)
    
    # Create output directory
    output_dir = create_visualization_dir(output_dir)
    
    # Create visualizations
    create_similarity_histogram(df, output_dir)
    create_length_comparison(df, output_dir)
    create_ratio_similarity_plot(df, output_dir)
    create_top_bottom_analysis(df, output_dir)
    
    if has_clusters:
        create_cluster_visualization(df, output_dir)
    
    # Create summary report
    create_summary_report(df, output_dir, has_clusters)
    
    print(f"\nVisualization complete! All results saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create visualizations for interview analysis results')
    parser.add_argument('input_file', help='Path to the individual_analysis.csv file')
    parser.add_argument('--output', default='visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_interviews(args.input_file, args.output)