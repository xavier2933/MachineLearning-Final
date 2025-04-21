import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import re
import argparse
import time
from datetime import datetime


class BERTInterviewAnalyzer:
    def __init__(self, model_name='bert-base-uncased'):
        print("Initializing BERT model and tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
    def load_transcripts(self, csv_path):
        print(f"Loading transcripts from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['id', 'transcript']
        
        print(f"Loaded {len(df)} transcripts.")
        return df
    
    def parse_transcript(self, df):
        print("Parsing transcripts...")
        
        SPLIT_DELIMITER = '|'
        
        df['interviewer_text'] = ""
        df['interviewee_text'] = ""
        
        for transcript_index, row in df.iterrows():
            transcript = row['transcript']
            parts = transcript.split(SPLIT_DELIMITER)
            
            interviewer_parts = []
            interviewee_parts = []
            
            for part in parts:
                if part.strip().startswith("Interviewer:"):
                    interviewer_parts.append(part.replace("Interviewer:", "").strip())
                elif part.strip().startswith("Interviewee:"):
                    interviewee_parts.append(part.replace("Interviewee:", "").strip())
            
            interviewer_text = " ".join(interviewer_parts)
            interviewee_text = " ".join(interviewee_parts)
            
            df.at[transcript_index, 'interviewer_text'] = interviewer_text
            df.at[transcript_index, 'interviewee_text'] = interviewee_text
        
        return df
    
    def split_into_chunks(self, text, max_length=128):
        """Chunks to fit max size"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(tokens)
            
            # If adding sentence exceeds max_length, add current chunk to chunks and start new chunk
            if current_length + sentence_length > max_length - 2:  # -2 for [CLS] and [SEP]
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_bert_embeddings(self, text):
        if not text or len(text.strip()) == 0:
            # Return a tensor of zeros with the correct dimensions
            return torch.zeros(768)
        
        # Handle long text by chunking
        if len(self.tokenizer.tokenize(text)) > 126:  # 128 - 2 for [CLS] and [SEP]
            chunks = self.split_into_chunks(text)

            embeddings = []
            for chunk in chunks:
                chunk_embedding = self._get_single_embedding(chunk)
                embeddings.append(chunk_embedding)
            return torch.mean(torch.stack(embeddings), dim=0)
        else:
            return self._get_single_embedding(text)
    
    def _get_single_embedding(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        
        if len(tokenized_text) < 2:
            tokenized_text = ["[CLS]", "[SEP]"]
            
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        
        segments_ids = [0] * len(indexed_tokens)
        
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
        
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        
        # Use the output from the last layer for the [CLS] token
        sentence_embedding = encoded_layers[-1][0][0].cpu()
        return sentence_embedding
    
    def analyze_transcript_collection(self, csv_path, output_dir='results'):
        start_time = time.time()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        transcripts_df = self.load_transcripts(csv_path)
        parsed_df = self.parse_transcript(transcripts_df)
        
        parsed_df['interviewer_embedding'] = None
        parsed_df['interviewee_embedding'] = None
        
        print("Generating BERT embeddings for all transcripts...")
        for idx, row in parsed_df.iterrows():
            if idx % 10 == 0 and idx > 0:
                print(f"Processed {idx}/{len(parsed_df)} transcripts...")
            
            interviewer_text = row['interviewer_text']
            interviewee_text = row['interviewee_text']
            
            interviewer_embedding = self.get_bert_embeddings(interviewer_text)
            interviewee_embedding = self.get_bert_embeddings(interviewee_text)
            
            parsed_df.at[idx, 'interviewer_embedding'] = interviewer_embedding
            parsed_df.at[idx, 'interviewee_embedding'] = interviewee_embedding
        
        print("\nAnalyzing individual transcripts...")
        individual_results = []
        
        for idx, row in parsed_df.iterrows():
            transcript_id = row['id']
            interviewer_embedding = row['interviewer_embedding']
            interviewee_embedding = row['interviewee_embedding']
            
            # Calculate similarity
            if isinstance(interviewer_embedding, torch.Tensor) and isinstance(interviewee_embedding, torch.Tensor):
                similarity = torch.nn.functional.cosine_similarity(
                    interviewer_embedding.unsqueeze(0),
                    interviewee_embedding.unsqueeze(0)
                ).item()
            else:
                similarity = None
            
            interviewer_length = len(row['interviewer_text'])
            interviewee_length = len(row['interviewee_text'])
            
            individual_results.append({
                'id': transcript_id,
                'similarity': similarity,
                'interviewer_length': interviewer_length,
                'interviewee_length': interviewee_length,
                'ratio': interviewee_length / interviewer_length if interviewer_length > 0 else None
            })
        
        individual_df = pd.DataFrame(individual_results)
        
        individual_df.to_csv(f"{output_dir}/individual_analysis.csv", index=False)
        
        all_interviewer_embeddings = []
        all_interviewee_embeddings = []
        
        for idx, row in parsed_df.iterrows():
            if isinstance(row['interviewer_embedding'], torch.Tensor):
                all_interviewer_embeddings.append(row['interviewer_embedding'])
            if isinstance(row['interviewee_embedding'], torch.Tensor):
                all_interviewee_embeddings.append(row['interviewee_embedding'])
        
        if len(all_interviewer_embeddings) > 0 and len(all_interviewee_embeddings) > 0:
            interviewer_stack = torch.stack(all_interviewer_embeddings)
            interviewee_stack = torch.stack(all_interviewee_embeddings)
            
            global_interviewer_mean = torch.mean(interviewer_stack, dim=0)
            global_interviewee_mean = torch.mean(interviewee_stack, dim=0)
            
            global_similarity = torch.nn.functional.cosine_similarity(
                global_interviewer_mean.unsqueeze(0),
                global_interviewee_mean.unsqueeze(0)
            ).item()
            
            # Perform PCA on all embeddings combined
            all_embeddings = torch.cat([interviewer_stack, interviewee_stack], dim=0)
            all_embeddings_np = all_embeddings.numpy()
            
            # Create labels for PCA visualization
            labels = ['Interviewer'] * len(interviewer_stack) + ['Interviewee'] * len(interviewee_stack)
            
            # Apply PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(all_embeddings_np)
            
            # Create a DataFrame for visualization
            vis_df = pd.DataFrame({
                'pca_1': reduced_embeddings[:, 0],
                'pca_2': reduced_embeddings[:, 1],
                'role': labels
            })
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            sns.scatterplot(x='pca_1', y='pca_2', hue='role', data=vis_df)
            plt.title('PCA of All Interview Embeddings')
            plt.savefig(f"{output_dir}/global_pca_visualization.png")
            
            # Perform clustering on interviewee embeddings
            if len(interviewee_stack) >= 5:  # Only cluster if we have enough samples
                num_clusters = min(5, len(interviewee_stack))
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                interviewee_np = interviewee_stack.numpy()
                clusters = kmeans.fit_predict(interviewee_np)
                
                # Find representative examples for each cluster
                cluster_results = []
                
                for cluster_id in range(num_clusters):
                    # Get indices of examples in this cluster
                    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                    
                    if cluster_indices:
                        # Get distances to centroid
                        centroid = kmeans.cluster_centers_[cluster_id]
                        distances = [np.linalg.norm(interviewee_np[idx] - centroid) for idx in cluster_indices]
                        
                        # Find closest examples to centroid
                        sorted_indices = np.argsort(distances)
                        closest_indices = [cluster_indices[i] for i in sorted_indices[:3]]
                        
                        # Get transcript IDs for closest examples
                        closest_ids = [parsed_df.iloc[idx]['id'] for idx in closest_indices]
                        
                        cluster_results.append({
                            'cluster_id': cluster_id,
                            'size': len(cluster_indices),
                            'representative_ids': closest_ids
                        })
                
                # Save cluster results
                cluster_df = pd.DataFrame(cluster_results)
                cluster_df.to_csv(f"{output_dir}/cluster_analysis.csv", index=False)
                
                # Add cluster information to individual results
                for i, cluster_id in enumerate(clusters):
                    individual_df.loc[individual_df['id'] == parsed_df.iloc[i]['id'], 'cluster'] = cluster_id
                
                # Update individual results file
                individual_df.to_csv(f"{output_dir}/individual_analysis.csv", index=False)

        # Create a summary report
        with open(f"{output_dir}/analysis_summary.txt", 'w') as f:
            f.write(f"BERT Interview Analysis Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input File: {csv_path}\n")
            f.write(f"Number of Transcripts: {len(parsed_df)}\n\n")
            
            if len(all_interviewer_embeddings) > 0 and len(all_interviewee_embeddings) > 0:
                f.write(f"Global Interviewer-Interviewee Similarity: {global_similarity:.4f}\n\n")
            
            f.write(f"Individual Transcript Statistics:\n")
            f.write(f"  Average Similarity: {individual_df['similarity'].mean():.4f}\n")
            f.write(f"  Min Similarity: {individual_df['similarity'].min():.4f}\n")
            f.write(f"  Max Similarity: {individual_df['similarity'].max():.4f}\n\n")
            
            f.write(f"Text Length Statistics:\n")
            f.write(f"  Average Interviewer Length: {individual_df['interviewer_length'].mean():.1f} characters\n")
            f.write(f"  Average Interviewee Length: {individual_df['interviewee_length'].mean():.1f} characters\n")
            f.write(f"  Average Interviewee/Interviewer Ratio: {individual_df['ratio'].mean():.2f}\n\n")
            
            if 'cluster' in individual_df.columns:
                f.write(f"Cluster Information:\n")
                for cluster_id in range(cluster_df['cluster_id'].max() + 1):
                    cluster_size = cluster_df.loc[cluster_df['cluster_id'] == cluster_id, 'size'].values[0]
                    f.write(f"  Cluster {cluster_id}: {cluster_size} transcripts\n")
            
            elapsed_time = time.time() - start_time
            f.write(f"\nAnalysis completed in {elapsed_time:.2f} seconds.")
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")
        print(f"Analysis took {time.time() - start_time:.2f} seconds.")
        
        return {
            'individual_df': individual_df,
            'output_dir': output_dir
        }


def get_clustered_analysis(csv_path, output_path):

    analyzer = BERTInterviewAnalyzer()
    result = analyzer.analyze_transcript_collection(csv_path, output_path)

    return result['individual_df']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze interview transcripts using BERT embeddings')
    parser.add_argument('csv_file', help='Path to the CSV file containing interview transcripts')
    parser.add_argument('--output', default='results', help='Output directory for analysis results')

    args = parser.parse_args()
    get_clustered_analysis(args.csv_file, args.output)