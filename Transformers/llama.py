import ollama
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import re
from scipy.stats import pearsonr
import time

class Llama:
    def __init__(self):
        self.model = 'llama3.2:1b'
        self.temperature = 0.7
        self.max_tokens = 200
        self.transcript_path = '../RawProvidedData/MITInterview/transcripts.csv'
        self.scores_path = '../RawProvidedData/MITInterview/scores.csv'
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            self.content = file.read()

    def load_transcripts(self, csv_path) -> pd.DataFrame:
        """From DataExtractor"""
        print(f"Loading transcripts from {csv_path}...")
        df = pd.read_csv(csv_path)

        def take_only_two_columns(df: pd.DataFrame) -> pd.DataFrame:
            if len(df.columns) >= 2:
                df = df.iloc[:, :2]
                df.columns = ['id', 'transcript']
            return df

        df = take_only_two_columns(df)
        print(f"Loaded {len(df)} transcripts.")
        return df
    
    def load_actual_scores(self, scores_path) -> pd.DataFrame:
        print(f"Loading actual scores from {scores_path}...")
        scores_df = pd.read_csv(scores_path)
        print(f"Loaded {len(scores_df)} score records.")
        return scores_df
    
    def analyze_transcript_llama(self, transcript):
        # prompt = self.content + transcript + '\nQuestion: Rate this interview on a scale of 1-7, '
        # 'where 1 is the lowest score and 7 is the highest score.'
        # ' Start your answer with just the numeric score (1-7) followed by your reasoning.'
        # ' Try to factor in perceived interviewer excitement into your score'
        prompt = transcript + '\nQuestion: Rate this interview on a scale of 1-7, '
        'where 1 is the lowest score and 7 is the highest score.'
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            }
        )
        return response['response']
    
    # Claude
    def extract_interview_score(self, analysis_text):
        # Pattern to extract a number 1-7 at the beginning of the text
        match = re.match(r'^\s*([1-7])[\s\.,:]', analysis_text)
        if match:
            return int(match.group(1))
        else:
            # If no match, try to find any mention of a score between 1-7 in the first sentence
            match = re.search(r'\b([1-7])\b', analysis_text.split('.')[0])
            if match:
                return int(match.group(1))
            else:
                return -1
            

    def evaluate_model_performance(self, results_path='transcript_score_analysis_results.csv'):
        from sklearn.preprocessing import MinMaxScaler
        
        predictions_df = pd.read_csv(results_path)
        
        actual_scores_df = self.load_actual_scores(self.scores_path)
        
        # rename column for merge
        actual_scores_df = actual_scores_df.rename(columns={"Participant": "id"})
        
        score_col = None
        for col in actual_scores_df.columns:
            if col != "id" and actual_scores_df[col].dtype in [np.int64, np.float64]:
                score_col = col
                break
        
        actual_scores_df = actual_scores_df.rename(columns={score_col: "actual_score"})
        
        merged_df = predictions_df.merge(actual_scores_df[['id', 'actual_score']], on='id', how='inner')
        print(f"Merged data contains {len(merged_df)} records.")
        
        print(f"Predicted scores range: {merged_df['predicted_score'].min()} to {merged_df['predicted_score'].max()}")
        print(f"Actual scores range: {merged_df['actual_score'].min()} to {merged_df['actual_score'].max()}")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        scores_df = pd.DataFrame({
            'predicted_score': merged_df['predicted_score'],
            'actual_score': merged_df['actual_score']
        })
        
        scaled_scores = scaler.fit_transform(scores_df)
        
        merged_df['predicted_score_norm'] = scaled_scores[:, 0]
        merged_df['actual_score_norm'] = scaled_scores[:, 1]
        
        merged_df['score_diff'] = abs(merged_df['predicted_score_norm'] - merged_df['actual_score_norm'])
        
        mae = merged_df['score_diff'].mean()
        
        rmse = np.sqrt(np.mean(np.square(merged_df['predicted_score_norm'] - merged_df['actual_score_norm'])))
        
        pearson_corr, p_value = pearsonr(merged_df['predicted_score_norm'], merged_df['actual_score_norm'])
        
        correlation = merged_df[['predicted_score_norm', 'actual_score_norm']].corr().iloc[0, 1]
        
        epsilon = 1e-10 # avoid /0 error
        merged_df['rel_error'] = merged_df['score_diff'] / (merged_df['actual_score_norm'] + epsilon)
        mean_are = merged_df['rel_error'].mean()
        
        print("\n--- Original Metrics ---")
        orig_mae = abs(merged_df['predicted_score'] - merged_df['actual_score']).mean()
        orig_rmse = np.sqrt(np.mean(np.square(merged_df['predicted_score'] - merged_df['actual_score'])))
        orig_corr, orig_p = pearsonr(merged_df['predicted_score'], merged_df['actual_score'])
        print(f"Original Mean Absolute Error: {orig_mae:.4f}")
        print(f"Original Root Mean Squared Error: {orig_rmse:.4f}")
        print(f"Original Pearson Correlation: {orig_corr:.4f} (p-value: {orig_p:.4f})")
        
        print("\n--- Normalized Metrics (0-1 scale) ---")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {p_value:.4f})")
        print(f"Mean Absolute Relative Error: {mean_are:.4f}")
        
        merged_df.to_csv('evaluation_results_normed_normal.csv', index=False)
        print("Evaluation results saved to evaluation_results.csv")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'pearson_correlation': pearson_corr,
            'p_value': p_value,
            'correlation': correlation,
            'mean_are': mean_are,
            'merged_data': merged_df
        }
        
    def run(self):
        df = self.load_transcripts(self.transcript_path)
        
        results = []
        df_size = len(df)
        for i, row in df.iterrows():
            id = row['id']
            text = row['transcript']
            # if i == 10:
            #     break
            print(f"Processing transcript {i+1}/{df_size}")

            try:
                res = self.analyze_transcript_llama(text)
                score = self.extract_interview_score(res)
                
                results.append({
                    'id': id,
                    'predicted_score': score,
                    'analysis': res
                })
                
            except Exception as e:
                print(f"Error processing transcript {id}: {str(e)}")

        results_df = pd.DataFrame(results)

        for idx, row in results_df.iterrows():
            print(f"Transcript ID: {row['id']}")
            print(f"Predicted Score: {row['predicted_score']}")
            print(f"Analysis: {row['analysis']}")
            print("-" * 50)

        results_df.to_csv('transcript_score_analysis_results_0shot.csv', index=False)
        print("Results saved to transcript_score_analysis_results_0shot.csv")
    

if __name__ == '__main__':
    llama = Llama()
    # llama.run() # uncomment to run
    llama.evaluate_model_performance()