import torch
import torch.nn.functional as F
import cv2
import math
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms

def cosine_sim(a, b):
    if not torch.is_tensor(a):
        a = torch.tensor(a)

    if not torch.is_tensor(b):
        b = torch.tensor(b)

    return F.cosine_similarity(a, b, dim=0).item()


def extract_embedding(face_img, model, device, inference_transform):
    face_tensor = inference_transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0)

def preprocess_face(face):
    face = cv2.resize(face, (112, 112))
    return face

def get_rotation_angle(left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def load_raw_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def recognize_face(embedding, database, threshold):
    best_name = "unknown"
    best_score = 0.0
    for name, db_embeddings in database.items():
        # handle both old single-embedding and new multi-embedding entries
        if db_embeddings.dim() == 1:
            db_embeddings = db_embeddings.unsqueeze(0)
        scores = F.cosine_similarity(
            embedding.unsqueeze(0).expand(db_embeddings.shape[0], -1),
            db_embeddings
        )
        score = scores.max().item()
        if score > best_score:
            best_score = score
            best_name = name
    if best_score < threshold:
        return "unknown", best_score
    return best_name, best_score


def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def crop_face(img, box, margin=0.2):
    x, y, w, h = box
    img_h, img_w = img.shape[:2]

    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)

    return img[y1:y2, x1:x2]

detector = MTCNN()

def detect_face(img):
    results = detector.detect_faces(img)
    if len(results) == 0:
        return None
    return results[0]  


def rotate_image(img, center, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated