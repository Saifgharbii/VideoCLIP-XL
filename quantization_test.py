from torch.quantization import quantize_dynamic, quantize_qat, prepare, convert
import torch
from utils.vision_encoder import get_vision_encoder

model_path = "videoclip_xl_vision_encoder.pt"

vision_model = get_vision_encoder()
vision_model.load_state_dict(torch.load(model_path, map_location="cpu"))


try:
    vision_model.eval()
    # Fake quantization setup
    vision_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(vision_model, inplace=True)
    torch.quantization.convert(vision_model, inplace=True)
    print("✅ Model supports quantization.")
    example_input = torch.randn(1, 3, 8, 224, 224) 
    model_traced = torch.jit.trace(vision_model, example_input)
    model_traced.save("vision_model_quantized_traced.pt")
except Exception as e:
    print("❌ Quantization failed:", e)