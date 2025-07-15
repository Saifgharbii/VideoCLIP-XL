import torch 
from modeling import VideoCLIP_XL

video_clip_xl = VideoCLIP_XL()
state_dict = torch.load("./VideoCLIP-XL.bin", map_location="cpu")
video_clip_xl.load_state_dict(state_dict)
video_clip_xl.cpu().eval()

def save_to_pt(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
    
if __name__ == "__main__":
    save_to_pt(video_clip_xl.text_model, "videoclip_xl_text_encoder.pt")
    print("Conversion to .pt format completed successfully.")
    save_to_pt(video_clip_xl.vision_model, "videoclip_xl_vision_encoder.pt")
    print("Conversion to .pt format completed successfully.")