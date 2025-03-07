import torch
import tiktoken
from utils.tokenizer import TiktokenTokenizer
from model.fftnet_model import FFTNet

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, device="cpu"):
    """
    Generate text using the FFTNet model.

    Args:
        model (FFTNet): Trained model
        tokenizer (TiktokenTokenizer): Tokenizer instance
        prompt (str): Text prompt to continue from
        max_length (int): Maximum length of generated text
        temperature (float): Temperature for sampling (higher = more random)
        device (str): Device to run inference on

    Returns:
        str: Generated text
    """
    model.eval()

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate tokens one by one
    for _ in range(max_length):
        with torch.no_grad():
            # Get model prediction
            outputs = model(input_ids)

            # Get logits for the next token (last position)
            next_token_logits = outputs[:, -1, :] / temperature

            # Sample from the distribution
            probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Append next token to input
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Break if we generate an EOS token
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated tokens
    generated_text = tokenizer.decode(input_ids[0])

    return generated_text

def main():
    # Model parameters
    vocab_size = 50257  # cl100k_base vocab size
    d_model = 128
    num_layers = 4

    # Initialize tokenizer
    tokenizer = TiktokenTokenizer(encoding_name="cl100k_base")

    # Initialize model
    model = FFTNet(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers
    )

    # Check if we have a trained model, otherwise just use untrained model for demo
    try:
        model.load_state_dict(torch.load("outputs/best_model.pth"))
        print("Loaded trained model")
    except:
        print("Using untrained model (for demonstration purposes only)")

    # Generate text from prompt
    prompt = "FFTNet is a neural network architecture that"
    generated_text = generate_text(model, tokenizer, prompt, max_length=50)

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
