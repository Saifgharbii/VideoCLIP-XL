from utils.text_encoder.simple_tokenizer import SimpleTokenizer, whitespace_clean

text_to_decode = [
    "Hello, world!",
    "This is a test sentence.",
    "I'm learning about tokenization.",
    "The quick brown fox jumps over the lazy dog.",
    "What's your favorite programming language?"
]

for text in text_to_decode:
    print(f"Original text: {text}")
    cleaned_text = whitespace_clean(text)
    print(f"Cleaned text: {cleaned_text}")
    
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.encode(cleaned_text)
    print(f"Tokens: {tokens}")
    
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded_text}\n")