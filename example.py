import torch
import argparse
import tiktoken
import os
from fftnet.utils.tokenizer import TiktokenTokenizer
from fftnet.model.fftnet_model import FFTNet

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=40, device="cpu"):
    """
    Generate text using the FFTNet model.

    Args:
        model (FFTNet): Trained model
        tokenizer (TiktokenTokenizer): Tokenizer instance
        prompt (str): Text prompt to continue from
        max_length (int): Maximum length of generated text
        temperature (float): Temperature for sampling (higher = more random)
        top_k (int): Number of highest probability tokens to consider for sampling
        device (str): Device to run inference on

    Returns:
        str: Generated text
    """
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Prompt tokens: {input_ids[0].tolist()}")
    print(f"Input shape: {input_ids.shape}")

    # Store the original prompt for output
    original_prompt = prompt

    # Track generation progress with tqdm if available
    try:
        from tqdm import tqdm
        gen_range = tqdm(range(max_length))
    except ImportError:
        gen_range = range(max_length)

    # Generate tokens one by one
    for step in gen_range:
        with torch.no_grad():
            # Make sure input sequence doesn't exceed model's max sequence length
            max_context = 1024  # Based on your model's max_seq_length
            if input_ids.size(1) > max_context:
                input_ids = input_ids[:, -max_context:]

            # Get model prediction
            try:
                outputs = model(input_ids)
            except Exception as e:
                print(f"\nError during inference: {e}")
                print(f"Input shape: {input_ids.shape}")
                print(f"Model expected max_seq_length: {max_context}")
                raise

            # Get logits for the next token (last position)
            next_token_logits = outputs[:, -1, :]

            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Debug: Print raw logits and probabilities for the first few steps
            if step < 2:
                print(f"\nStep {step}: Top 5 logits: {torch.topk(next_token_logits[0], 5).values.tolist()}")
                print(f"Step {step}: Top 5 token ids: {torch.topk(next_token_logits[0], 5).indices.tolist()}")

            # Apply top-k filtering
            if top_k > 0:
                # Keep only the top-k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)

                # Create a mask with zeros everywhere except for the top-k positions
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probabilities, num_samples=1)

            if step < 2:
                token_id = next_token[0].item()
                token_text = tokenizer.decode([token_id])
                print(f"Step {step}: Sampled token id: {token_id}, token text: '{token_text}'")
                probability = probabilities[0, token_id].item()
                print(f"Step {step}: Token probability: {probability:.6f}")

            # Append next token to input
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Break if we generate an EOS token
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                print("Reached EOS token, stopping generation.")
                break

    # Decode the generated tokens
    generated_text = tokenizer.decode(input_ids[0])

    # For cleaner output, separate the prompt from the generated text
    if len(original_prompt) < len(generated_text):
        continuation = generated_text[len(original_prompt):]
        return f"Prompt: {original_prompt}\nGenerated: {continuation}"
    else:
        return f"Prompt: {original_prompt}\nGenerated: {generated_text}"

def parse_args():
    parser = argparse.ArgumentParser(description="Text generation example using FFTNet model")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of FFTNet layers")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512, help="MLP hidden dimension")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--prompt", type=str, default="Hello", help="Text prompt")
    parser.add_argument("--device", type=str, default="mps", help="Device to run inference on")
    parser.add_argument("--tokenizer", type=str, default="cl100k_base", help="Tokenizer encoding name")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pth", help="Path to model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # First, try to load checkpoint to get model parameters
    model_config = {}
    try:
        model_path = args.model_path
        print(f"Loading model configuration from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device(args.device), weights_only=False)
        
        # Try to extract model configuration if available
        if isinstance(checkpoint, dict):
            # Store original values as fallback
            model_config['d_model'] = args.d_model
            model_config['num_layers'] = args.num_layers
            model_config['mlp_hidden_dim'] = args.mlp_hidden_dim
            model_config['max_seq_length'] = args.max_seq_length
            
            # Look for model parameters in checkpoint to extract dimensions
            if 'model_state_dict' in checkpoint:
                # Extract dimensions from state dict keys
                state_dict = checkpoint['model_state_dict']
                
                # Extract max_seq_length from pos_encoding
                if 'pos_encoding' in state_dict:
                    pos_shape = state_dict['pos_encoding'].shape
                    if len(pos_shape) == 3:
                        model_config['max_seq_length'] = pos_shape[1]
                        print(f"  - Extracted max_seq_length: {model_config['max_seq_length']}")
                
                # Extract mlp_hidden_dim from first layer's mlp weights
                for key in state_dict.keys():
                    if 'adaptive_filter.mlp.0.weight' in key:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) == 2:
                            model_config['mlp_hidden_dim'] = weight_shape[0]
                            print(f"  - Extracted mlp_hidden_dim: {model_config['mlp_hidden_dim']}")
                        break
    except Exception as e:
        print(f"Could not pre-load model configuration: {e}")
        # Will use command-line defaults

    # Model parameters - use extracted config or defaults
    d_model = model_config.get('d_model', args.d_model)
    num_layers = model_config.get('num_layers', args.num_layers)
    mlp_hidden_dim = model_config.get('mlp_hidden_dim', args.mlp_hidden_dim)
    max_seq_length = model_config.get('max_seq_length', args.max_seq_length)

    # Initialize tokenizer
    tokenizer = TiktokenTokenizer(encoding_name=args.tokenizer)
    print(f"Initialized tokenizer with vocab_size: {tokenizer.vocab_size}")

    # Print the special tokens if available
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        print(f"BOS token ID: {tokenizer.bos_token_id}")
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        print(f"EOS token ID: {tokenizer.eos_token_id}")

    # Initialize model
    model = FFTNet(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        max_seq_length=max_seq_length
    )
    
    print(f"Model initialized with:")
    print(f"  - d_model: {d_model}")
    print(f"  - num_layers: {num_layers}")
    print(f"  - mlp_hidden_dim: {mlp_hidden_dim}")
    print(f"  - max_seq_length: {max_seq_length}")

    # Move model to device
    device = args.device
    model.to(device)

    # Load model checkpoint
    try:
        model_path = args.model_path
        checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"Loaded trained model state from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded trained model state directly")

        print(f"Successfully loaded trained model from {model_path}")
    except Exception as e:
        print(f"WARNING: Using untrained model (for demonstration purposes only)")
        print(f"Could not load model from {model_path}. Error: {e}")

    # Print model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Generate text from prompt
    prompt = args.prompt
    print(f"\nGenerating text with:")
    print(f"- Temperature: {args.temperature}")
    print(f"- Top-k: {args.top_k}")
    print(f"- Max length: {args.max_length}")

    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )

    # Print the result
    print("\n" + "="*50)
    print(generated_text)
    print("="*50)

if __name__ == "__main__":
    main()
