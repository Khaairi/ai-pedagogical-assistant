from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from retinaface import RetinaFace
from huggingface_hub import hf_hub_download
from model import ViTMobilenet

# Initialize App
app = FastAPI()

# Global variables to hold model
model = None
config = {}

def load_model():
    """Load model once during startup"""
    global model, config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = {
        "DEVICE": device,
        "IMG_SIZE": 224,
        "NUM_CLASSES": 7,
        "emotion_dict": {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"},
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    }

    # Initialize Architecture
    model = ViTMobilenet(
        num_classes=config["NUM_CLASSES"],
        in_channels=3,
        num_heads=12,
        embedding_dim=768,
        num_transformer_layers=12,
        mlp_size=3072
    )

    # Download and Load Weights
    print("Loading Model Weights...")
    repo_id = "MoKhaa/Hybrid_MobileNetV3_ViT"
    filename = "hybrid_mobilenet_vit_pooling_SAM_best.pt"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    print("Model Loaded Successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    try:
        faces = RetinaFace.detect_faces(image_np)
    except:
        return {"results": []}

    results = []
    
    if not isinstance(faces, dict):
        return {"results": []}

    face_tensors = []
    boxes = []
    
    for face_key, face_data in faces.items():
        x1, y1, x2, y2 = face_data["facial_area"]

        clean_box = [int(x1), int(y1), int(x2), int(y2)]
        # Crop
        face_crop = image.crop((x1, y1, x2, y2))
        
        # Transform for ViT
        tensor = config["transform"](face_crop)
        face_tensors.append(tensor)
        boxes.append(clean_box)

    if not face_tensors:
        return {"results": []}

    batch_tensor = torch.stack(face_tensors).to(config["DEVICE"])
    
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        class_ids = torch.argmax(probs, dim=1).cpu().numpy()
        scores = torch.max(probs, dim=1).values.cpu().numpy()

    for i, box in enumerate(boxes):
        label = config["emotion_dict"].get(int(class_ids[i]), "Unknown")
        results.append({
            "box": box,
            "label": label,
            "score": float(scores[i])
        })

    return {"results": results}
