import torch
import cv2
import numpy as np
from torchvision import transforms

from backend.config import *

from utils.helper import *


# Load model and database once
model = None
database = None
device = None
inference_transform = None

register_name = None
register_embeddings = []

def load_resources():
    global model, database, device, inference_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()

    database = torch.load(DATABASE_PATH, map_location=device)

    inference_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    print("Model and database loaded")

def recognize_from_image(image_path):

    img = load_raw_image(image_path)

    face_data = detect_face(img)

    if face_data is None:
        return "No face detected", 0.0

    box = face_data["box"]
    keypoints = face_data["keypoints"]

    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]

    angle = get_rotation_angle(left_eye, right_eye)

    center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2),
    )

    aligned = rotate_image(img, center, angle)

    face = crop_face(aligned, box)

    face = preprocess_face(face)

    embedding = extract_embedding(face, model, device, inference_transform)

    name, score = recognize_face(embedding, database, THRESHOLD)

    return name, float(score)

def register_start(name: str):

    global register_name, register_embeddings

    register_name = name
    register_embeddings = []

    return {"status": "started", "name": name}

def register_frame(image, current_name):
    global register_embeddings

    # Same pipeline as recognize_from_image
    face_data = detect_face(image)

    if face_data is None:
        return {"status": "no_face", "count": len(register_embeddings)}

    box = face_data["box"]
    keypoints = face_data["keypoints"]

    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]

    angle = get_rotation_angle(left_eye, right_eye)

    center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2),
    )

    aligned = rotate_image(image, center, angle)
    face = crop_face(aligned, box)
    face = preprocess_face(face)
    embedding = extract_embedding(face, model, device, inference_transform)

    # Check pose variation before accepting frame
    if len(register_embeddings) > 0:
        similarities = [
            F.cosine_similarity(embedding.unsqueeze(0), e.unsqueeze(0)).item()
            for e in register_embeddings
        ]
        if max(similarities) > 0.98:  # too similar to existing frame
            return {"status": "move_face", "count": len(register_embeddings)}

    register_embeddings.append(embedding)

    # Collected enough frames → save to database
    if len(register_embeddings) >= 150:
        stacked = torch.stack(register_embeddings)
        database[current_name] = stacked
        torch.save(database, DATABASE_PATH)
        register_embeddings = []
        return {"status": "ok", "count": 150}

    return {"status": "captured", "count": len(register_embeddings)}

def register_save():

    global register_name, register_embeddings, database

    if register_name is None:
        return {"status": "no_session"}

    if len(register_embeddings) < REGISTER_MIN_IMAGES:
        return {
            "status": "not_enough",
            "count": len(register_embeddings),
        }

    database[register_name] = torch.stack(register_embeddings)

    torch.save(database, DATABASE_PATH)

    name = register_name
    count = len(register_embeddings)

    register_name = None
    register_embeddings = []

    return {
        "status": "saved",
        "name": name,
        "count": count,
    }

