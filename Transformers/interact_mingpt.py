import torch
import os
import argparse

def load_and_interact(model_dir):
    """Load a trained minGPT model and interact with it"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if files exist
    vocab_path = os.path.join(model_dir, "vocab.pt")
    model_path = os.path.join(model_dir, "model.pt")
    
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found at {vocab_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        # Load vocabulary
        print("Loading vocabulary...")
        vocab_data = torch.load(vocab_path, map_location=device)
        stoi = vocab_data['stoi']
        itos = vocab_data['itos']
        vocab_size = vocab_data['vocab_size']
        print(f"Vocabulary loaded with {vocab_size} characters")
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Import minGPT modules
        from mingpt.model import GPT
        
        # Create model
        try:
            # If model_config is saved in the checkpoint
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                print("Using saved model configuration")
            else:
                # Create default config (adjust if needed)
                model_config = GPT.get_default_config()
                model_config.vocab_size = vocab_size
                model_config.block_size = 128  # Adjust if needed
                print("Using default model configuration")
                
            model = GPT(model_config)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            print("Model loaded successfully!")
            
            # Interactive loop
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
                
                # Make sure all characters in prompt are in vocabulary
                safe_prompt = ''.join([c for c in prompt if c in stoi])
                if len(safe_prompt) < len(prompt):
                    print(f"\nWarning: Some characters in prompt are not in vocabulary. Using '{safe_prompt}' instead.")
                
                # Truncate prompt if needed
                block_size = model_config.block_size
                if len(safe_prompt) > block_size - 10:  # Leave room for generation
                    safe_prompt = safe_prompt[-(block_size - 10):]
                    print(f"\nPrompt truncated to fit in model's context window: '{safe_prompt}'")
                
                # Encode and generate
                try:
                    # Encode prompt
                    context = torch.tensor([stoi[c] for c in safe_prompt], dtype=torch.long, device=device).unsqueeze(0)
                    
                    # Generate tokens
                    generated = []
                    with torch.no_grad():
                        for _ in range(max_tokens):
                            # Forward pass
                            logits, _ = model(context)
                            
                            # Get predictions for last token
                            logits = logits[:, -1, :] / temperature
                            probs = torch.softmax(logits, dim=-1)
                            
                            # Sample
                            next_token = torch.multinomial(probs, num_samples=1)
                            
                            # Add to generated tokens
                            generated.append(next_token.item())
                            
                            # Update context
                            context = torch.cat([context, next_token], dim=1)
                            
                            # Don't exceed context size
                            if context.size(1) >= block_size:
                                context = context[:, -block_size:]
                    
                    # Convert to text
                    generated_text = safe_prompt + ''.join([itos[i] for i in generated])
                    
                    print("\rGenerated text:\n")
                    print(generated_text)
                    
                except Exception as e:
                    print(f"\nError during generation: {e}")
                    
        except Exception as e:
            print(f"Error creating/loading model: {e}")
            
    except Exception as e:
        print(f"Error loading files: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interact with a trained minGPT model')
    parser.add_argument('--model_dir', type=str, default='model_output',
                        help='Directory where model and vocabulary are saved')
    
    args = parser.parse_args()
    load_and_interact(args.model_dir)