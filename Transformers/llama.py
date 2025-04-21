import ollama
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import re
import time

class Llama:
    def __init__(self):
        self.model = 'llama3.2:1b'
        self.temperature = 0.7
        self.max_tokens = 200
        self.transcript_path = '../RawProvidedData/MITInterview/transcripts.csv'

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
    
    def analyze_transcript_llama(self, transcript):
        prompt = transcript + '\nQuestion: Did the candidate get the job? Start your answer with either "Yes" or "No" followed by your reasoning.'
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
            }
        )
        return response['response']
    
    def extract_job_status(self, analysis_text):
        if analysis_text.lower().startswith('yes'):
            return 'Yes'
        elif analysis_text.lower().startswith('no'):
            return 'No'
        else:
            return 'Uncertain'
        
    
    def run(self):
        df = self.load_transcripts(self.transcript_path)
        results = []
        df_size = len(df)
        for i, row in df.iterrows():
            id = row['id']
            text = row['transcript']
            print(f"Processing transcript {i+1}/{df_size}")

            try:
                res = self.analyze_transcript_llama(text)
                job_offered = self.extract_job_status(res)
                
                results.append({
                    'id': id,
                    'job_offered': job_offered,
                    'analysis': res
                })
                
            except Exception as e:
                print(f"Error processing transcript {id}: {str(e)}")

        results_df = pd.DataFrame(results)

        for idx, row in results_df.iterrows():
            print(f"Transcript ID: {row['id']}")
            print(f"Job Offered: {row['job_offered']}")
            print(f"Analysis: {row['analysis']}")
            print("-" * 50)

        results_df.to_csv('transcript_job_analysis_results.csv', index=False)
        print("Results saved to transcript_job_analysis_results.csv")

if __name__ == '__main__':
    llama = Llama()
    llama.run()