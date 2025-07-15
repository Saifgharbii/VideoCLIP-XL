import torch
from utils.vision_encoder import VisionEncoder
import logging
from utils.vision_encoder.clip_vision import VisionTransformer

# Set up logging to see any warnings
logging.basicConfig(level=logging.INFO)

VIT_PARAMETERS = {
    "inputs_image_res": 224,
    "kernel_size": 1,
    "heads": 16,
    "layers": 24,
    "center": True,
    "video_input_num_frames": 8,
    "drop_path_rate": 0.1,
    "vision_width": 1024,
    "embed_dim": 768,
    "masking_prob": 0.9,
    "patch_size": 14,
    "dropout": 0.0,
}

def convert_to_torchscript():
    # Load the model
    vision_model : VisionTransformer = VisionTransformer(
        input_resolution=VIT_PARAMETERS["inputs_image_res"], patch_size=VIT_PARAMETERS["patch_size"],
        width=VIT_PARAMETERS["vision_width"], layers=VIT_PARAMETERS["layers"], heads=VIT_PARAMETERS["heads"], 
        output_dim=VIT_PARAMETERS["embed_dim"], kernel_size=VIT_PARAMETERS["kernel_size"], 
        drop_path=VIT_PARAMETERS["drop_path_rate"], num_frames=VIT_PARAMETERS["video_input_num_frames"], 
        dropout=VIT_PARAMETERS["dropout"]
    )
    state_dict : dict = torch.load("videoclip_xl_vision_encoder.pt", map_location="cpu")
    new_state_dict = {}
    state_dict.pop("temp",None)
    for k, v in state_dict.items():
        if k.startswith("vision_encoder."):
            new_key = k[len("vision_encoder."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v 
    state_dict.clear()
    vision_model.load_state_dict(new_state_dict)
    vision_model.eval()


    
    # Create example input tensor for tracing
    example_input = torch.randn(1, 3, 8, 224, 224) 

    try:
        with torch.no_grad():
            model_traced = torch.jit.trace(vision_model, example_input)
            
            # Test the traced model
            original_output = vision_model(example_input)
            traced_output = model_traced(example_input)
            
            if torch.allclose(original_output, traced_output, atol=1e-5):
                print("✓ Trace method successful - outputs match!")
                torch.jit.save(model_traced, 'traced_vision_model.pt')
                return model_traced
            else:
                print("✗ Trace method failed - outputs don't match")
                raise Exception("Trace outputs don't match")
                
    except Exception as e:
        print(f"✗ Trace method failed: {e}")
        return None

def test_converted_model(model_path, example_input):
    """Test the converted model"""
    try:
        print("start testing")
        # Load the scripted model
        loaded_model = torch.jit.load(model_path)
        loaded_model.eval()
        
        with torch.no_grad():
            output = loaded_model(example_input)
            print(f"✓ Successfully loaded and ran scripted model")
            print(f"Output shape: {output.shape}")
            return True
            
    except Exception as e:
        print(f"✗ Failed to load/run scripted model: {e}")
        return False

if __name__ == "__main__":
    # Convert model
    converted_model = convert_to_torchscript()
    print("Model conversion complete")
    
    if converted_model is not None:
        # Test with example input
        test_input = torch.randn(1, 3, 8, 224, 224)
        
        # Test the saved model
        model_path = 'scripted_vision_model.pt' if 'scripted' in str(type(converted_model)) else 'traced_vision_model.pt'
        success = test_converted_model(model_path, test_input)
        
        if success:
            print(f"\n✓ Model successfully converted and saved as {model_path}")
            print("This model should work with libtorch in C++")
        else:
            print("\n✗ Model conversion failed")
    else:
        print("\n✗ Could not convert model to TorchScript")