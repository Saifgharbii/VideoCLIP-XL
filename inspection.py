import torch

# Load the state dictionary
state_dict = torch.load("VideoCLIP-XL.bin", map_location="cpu")

# Print layer names and shapes
print("Model Layers:")
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
    
