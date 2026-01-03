from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from retinaface import RetinaFace
import onnxruntime as ort

# Initialize App
app = FastAPI()

# Global variables to hold model
ort_session = None
config = {}

def load_model():
    """Load model once during startup"""
    global ort_session, config
    
    # Configuration
    config = {
        "IMG_SIZE": 224,
        "NUM_CLASSES": 7,
        "emotion_dict": {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"},
        "transform": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    }

    print("Loading ONNX Model...")
    # Providers: Try to use GPU if available, otherwise CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession("facial_expression_model.onnx", providers=providers)
    print("Active Providers:", ort_session.get_providers())
    print("ONNX Model Loaded!")

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
    
    import torch

    batch_tensor = torch.stack(face_tensors)

    # ONNX requires Numpy
    input_data = batch_tensor.numpy()

    # run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)

    logits = ort_outs[0]

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    probs = softmax(logits)
    class_ids = np.argmax(probs, axis=1)
    scores = np.max(probs, axis=1)

    for i, box in enumerate(boxes):
        label = config["emotion_dict"].get(int(class_ids[i]), "Unknown")
        results.append({
            "box": box,
            "label": label,
            "score": float(scores[i])
        })

    return {"results": results}
