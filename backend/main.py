from fastapi import FastAPI, UploadFile, File
import shutil
import os
from PIL import Image
from io import BytesIO
import numpy as np
from backend.recognition_pipeline import *

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup_event():
    load_resources()


@app.get("/")
def health_check():
    return {"message": "Face Recognition API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recognize")
async def recognize_face_api(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        name, score = recognize_from_image(file_path)
        return {"name": name, "confidence": score}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"name": "Error", "confidence": 0.0, "error": str(e)}

@app.post("/register")
async def register_identity(file: UploadFile = File(...), name: str = "unknown"):

    # placeholder for now
    return {
        "message": f"Register endpoint for {name} not implemented yet"
    }

@app.post("/register/start")
async def register_start_api(name: str):
    global current_session_name
    current_session_name = name
    return register_start(name)

@app.post("/register/frame")
async def register_frame_api(file: UploadFile = File(...)):
    global current_session_name

    if not current_session_name:
        return {"error": "No session started. Call /register/start first."}
    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image, dtype=np.uint8) 
    result = register_frame(image, current_session_name)

    return result

@app.post("/register/save")
async def register_save_api():

    return register_save()
