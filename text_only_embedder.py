import torch
import cv2
import numpy as np
from utils.text_encoder import text_encoder

model_path = "videoclip_xl_text_encoder.pt"
text_model = text_encoder.load()
text_model.load_state_dict(torch.load(model_path, map_location="cpu"))
text_model.eval().float()

def text_only_embedder(text: str) -> torch.Tensor:
    """
    Embed a text using the text encoder.

    Args:
        text (str): Input text to be embedded.

    Returns:
        torch.Tensor: Embedded text tensor.
    """
    text_inputs = text_encoder.tokenize([text], truncate=True).cpu()
    with torch.no_grad():
        text_emb = text_model.encode_text(text_inputs).float()
    return text_emb

if __name__ == "__main__":
    # Example usage
    text = "A beautiful sunset over the mountains."
    text_embedding = text_only_embedder(text)
    print("Text Embedding Shape:", text_embedding.shape)
    print("Text Embedding:", text_embedding)