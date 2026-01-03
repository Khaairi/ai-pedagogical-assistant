import torch
from model import ViTMobilenet  # Import your architecture
from huggingface_hub import hf_hub_download

def convert():
    device = torch.device('cpu')
    model = ViTMobilenet(
        num_classes=7,
        in_channels=3,
        num_heads=12,
        embedding_dim=768,
        num_transformer_layers=12,
        mlp_size=3072
    )

    print("Loading weights...")
    repo_id = "MoKhaa/Hybrid_MobileNetV3_ViT"
    filename = "hybrid_mobilenet_vit_pooling_SAM_best.pt"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    output_file = "facial_expression_model.onnx"
    print(f"Exporting to {output_file}...")
    
    torch.onnx.export(
        model,                      # The model
        dummy_input,                # The dummy input
        output_file,                # Output filename
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=13,           # Opset 12 is stable for Transformers/Attention
        do_constant_folding=True,   # Optimization: Pre-calculate constant values
        input_names=['input'],      # Name of the input layer
        output_names=['output'],    # Name of the output layer
        dynamic_axes={              # Allow variable batch sizes
            'input': {0: 'batch_size'},  
            'output': {0: 'batch_size'}
        }
    )
    print("Conversion Complete!")

if __name__ == "__main__":
    convert()