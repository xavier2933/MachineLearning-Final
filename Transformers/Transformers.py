import torch
from torch.utils.data import Dataset
import numpy as np
import multiprocessing
import os
import argparse
import re
import pandas as pd

BLOCK_SIZE = 512

class TextDataset(Dataset):
    """
    Dataset for training minGPT on text data
    """
    def __init__(self, text, block_size=BLOCK_SIZE, is_csv=False, csv_path=None):
        # If we're using a CSV file instead of raw text
        if is_csv and csv_path:
            text = self.load_from_csv(csv_path)
        
        # Create a character-level tokenizer for simplicity
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string to index
        self.itos = {i: ch for i, ch in enumerate(chars)}  # index to string
        
        # Tokenize the text
        data = [self.stoi[c] for c in text]
        
        # Create examples of sequence/target pairs for training
        self.examples = []
        for i in range(0, len(data) - block_size):
            x = data[i:i + block_size]
            y = data[i + 1:i + block_size + 1]
            self.examples.append((x, y))
            
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Number of examples: {len(self.examples)}")
    
    def load_from_csv(self, csv_path):
        """Load transcript data from a CSV file"""
        print(f"Loading transcripts from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Take only the first two columns and rename them
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['id', 'transcript']
        
        print(f"Loaded {len(df)} transcripts.")
        
        # Concatenate all transcripts into a single text
        all_text = " ".join(df['transcript'].tolist())
        return all_text
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        return ''.join([self.itos[i] for i in ids])
    
    def encode(self, text):
        """Convert text to token IDs"""
        return [self.stoi[c] for c in text]
    
    def save_vocab(self, path):
        """Save the vocabulary mappings to a file"""
        torch.save({
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size
        }, path)
    
    @classmethod
    def load_vocab(cls, path):
        """Load the vocabulary mappings from a file"""
        data = torch.load(path)
        dataset = cls.__new__(cls)
        dataset.stoi = data['stoi']
        dataset.itos = data['itos']
        dataset.vocab_size = data['vocab_size']
        return dataset

def train_model(text_data=None, csv_path=None, output_dir="model_output", block_size=BLOCK_SIZE, batch_size=32, 
                max_iters=1000, learning_rate=5e-4, 
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train a minGPT model on text data and save it"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the dataset
    if csv_path:
        train_dataset = TextDataset("", block_size=block_size, is_csv=True, csv_path=csv_path)
    else:
        train_dataset = TextDataset(text_data, block_size=block_size)
    
    train_dataset.save_vocab(os.path.join(output_dir, "vocab.pt"))
    
    # Import minGPT modules
    from mingpt.model import GPT
    
    # Create model config
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = train_dataset.vocab_size
    model_config.block_size = block_size
    
    # Create model
    model = GPT(model_config)
    model.to(device)
    
    # Import trainer
    from mingpt.trainer import Trainer
    
    # Create trainer config
    train_config = Trainer.get_default_config()
    train_config.learning_rate = learning_rate
    train_config.max_iters = max_iters
    train_config.batch_size = batch_size
    train_config.num_workers = min(4, multiprocessing.cpu_count())  # Use fewer workers to prevent issues
    
    # Create trainer
    trainer = Trainer(train_config, model, train_dataset)
    
    # Train model
    trainer.run()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'block_size': block_size
    }, os.path.join(output_dir, "model.pt"))
    
    return model, train_dataset

def generate_text(model, dataset, prompt="", max_tokens=100, temperature=1.0, 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
    """Generate text using a trained minGPT model"""
    model.eval()
    
    # Convert prompt to token IDs
    if prompt:
        # Make sure all characters in prompt are in the vocabulary
        safe_prompt = ''.join([c for c in prompt if c in dataset.stoi])
        if len(safe_prompt) < len(prompt):
            print(f"Warning: Some characters in prompt are not in the vocabulary. Using '{safe_prompt}' instead.")
        context = torch.tensor(dataset.encode(safe_prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate tokens
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            logits, _ = model(context)
            
            # Focus on the last token's predictions
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens
            generated.append(next_token.item())
            
            # Append the new token to the context
            context = torch.cat([context, next_token], dim=1)
    
    # Combine prompt and generated text
    if prompt:
        return prompt + dataset.decode(generated)
    else:
        return dataset.decode(generated)

def load_model_for_generation(model_dir="your_model_dir", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained minGPT model and vocabulary for text generation"""
    # Load vocabulary
    dataset = TextDataset.load_vocab(os.path.join(model_dir, "vocab.pt"))
    print("Loaded dataset")
    
    # Import minGPT modules
    from mingpt.model import GPT
    
    # Load saved model data
    checkpoint = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    print("Loaded model")
    
    # Get a fresh default config
    model_config = GPT.get_default_config()
    
    # Set the essential parameters based on the checkpoint
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = dataset.vocab_size  # Use the vocab size from the loaded dataset
    model_config.block_size = checkpoint['block_size']  # Use the block_size from the checkpoint
    
    # Create model with the config
    model = GPT(model_config)
    print('Model created successfully')
    
    # Load saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("loaded saved weights")
    
    model.to(device)
    model.eval()
    
    return model, dataset

def interactive_interface(model_dir="model_output"):
    """Interactive interface for text generation with the trained model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} using {device}...")
    
    try:
        model, dataset = load_model_for_generation(model_dir, device)
        
        print("\n" + "="*50)
        print(" MinGPT Text Generation Interface ")
        print("="*50)
        print("Type 'exit' to quit, 'help' for options\n")
        
        temperature = 1.0
        max_tokens = 100
        
        while True:
            prompt = input("\nPrompt> ")
            if prompt.lower() == 'exit':
                break
            elif prompt.lower() == 'help':
                print("\nCommands:")
                print("  exit - Exit the program")
                print("  help - Show this help message")
                print("  temp=X - Set temperature to X (e.g., temp=0.8)")
                print("  max=X - Set maximum tokens to X (e.g., max=200)")
                print("  settings - Show current settings")
                continue
            elif prompt.lower() == 'settings':
                print(f"\nCurrent settings:")
                print(f"  Temperature: {temperature}")
                print(f"  Max tokens: {max_tokens}")
                continue
            elif prompt.lower().startswith('temp='):
                try:
                    temperature = float(prompt.split('=')[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Invalid temperature format. Use 'temp=0.8' for example.")
                continue
            elif prompt.lower().startswith('max='):
                try:
                    max_tokens = int(prompt.split('=')[1])
                    print(f"Max tokens set to {max_tokens}")
                except:
                    print("Invalid max tokens format. Use 'max=200' for example.")
                continue
            
            # Generate text
            print("\nGenerating...", end="", flush=True)
            generated_text = generate_text(
                model, 
                dataset, 
                prompt=prompt, 
                max_tokens=max_tokens, 
                temperature=temperature,
                device=device
            )
            print("\rGenerated text:\n")
            print(generated_text)
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train and interact with minGPT on text data')
    parser.add_argument('--mode', type=str, choices=['train', 'interact'], required=True,
                        help='Mode: train a new model or interact with an existing one')
    parser.add_argument('--input_file', type=str, help='Text file or CSV file for training (required for train mode)')
    parser.add_argument('--csv', action='store_true', help='Specify if the input file is a CSV')
    parser.add_argument('--model_dir', type=str, default='model_output',
                        help='Directory to save or load the model')
    parser.add_argument('--block_size', type=int, default=BLOCK_SIZE,
                        help='Context size for the model (sequence length)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Check if input file is provided
        if not args.input_file:
            print("Error: --input_file is required for train mode")
            return
        
        # Train model
        print(f"Training model with block_size={args.block_size}, batch_size={args.batch_size}")
        
        if args.csv:
            # Train from CSV file
            model, dataset = train_model(
                csv_path=args.input_file,
                output_dir=args.model_dir,
                block_size=args.block_size,
                batch_size=args.batch_size,
                max_iters=args.iterations,
                learning_rate=args.learning_rate,
                device=device
            )
            print(f"Training complete. Model saved to {args.model_dir}")
            
            # For sample generation, we'll need some text from the CSV
            try:
                df = pd.read_csv(args.input_file)
                if len(df) > 0:
                    sample_text = df.iloc[0, 1]  # Get first transcript
                    sample_prompt = sample_text[:20] if len(sample_text) > 20 else sample_text[:len(sample_text)//2]
                    print("\nGenerating sample text with prompt:", sample_prompt)
                    sample_text = generate_text(model, dataset, prompt=sample_prompt, max_tokens=100, device=device)
                    print(sample_text)
            except Exception as e:
                print(f"Error generating sample: {e}")
                
        else:
            # Load text data from regular text file
            try:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    text_data = f.read()
                    
                print(f"Loaded {len(text_data)} characters from {args.input_file}")
                
                # Train model with text data
                model, dataset = train_model(
                    text_data=text_data, 
                    output_dir=args.model_dir,
                    block_size=args.block_size,
                    batch_size=args.batch_size,
                    max_iters=args.iterations,
                    learning_rate=args.learning_rate,
                    device=device
                )
                
                print(f"Training complete. Model saved to {args.model_dir}")
                
                # Generate sample text
                sample_prompt = text_data[:20] if len(text_data) > 20 else text_data[:len(text_data)//2]
                print("\nGenerating sample text with prompt:", sample_prompt)
                sample_text = generate_text(model, dataset, prompt=sample_prompt, max_tokens=100, device=device)
                print(sample_text)
                
            except Exception as e:
                print(f"Error: {e}")
            
    elif args.mode == 'interact':
        # Launch interactive interface
        interactive_interface(model_dir=args.model_dir)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()