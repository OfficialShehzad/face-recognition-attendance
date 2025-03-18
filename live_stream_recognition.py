import os
import cv2
import numpy as np
from collections import defaultdict

from movement_tracker import track_movement

# Paths to ONNX models
DETECTION_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"

# Load Face Detector (YuNet)
detector = cv2.FaceDetectorYN.create(DETECTION_MODEL, "", (320, 320))

# Load Face Recognizer (SFace)
recognizer = cv2.FaceRecognizerSF.create(RECOGNITION_MODEL, "")

# Load stored embeddings
known_embeddings = {}
for file in os.listdir("face_embeddings"):
    if file.endswith(".npy"):
        name = os.path.splitext(file)[0]  # Extract file name
        known_embeddings[name] = np.load(os.path.join("face_embeddings", file))

# Flatten embeddings before computing cosine similarity
def cosine_similarity(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_faces(frame, get_db):
    """Detect and recognize faces in a frame."""
    height, width, _ = frame.shape
    detector.setInputSize((width, height))

    # Detect Faces
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            center_x, center_y = x + w // 2, y + h // 2  # Get face center

            # Align & Extract Features
            aligned_face = recognizer.alignCrop(frame, face)
            face_embedding = recognizer.feature(aligned_face)

            # Compare with stored embeddings
            recognized_name = "Unknown"
            max_similarity = 0.0
            threshold = 0.363  # Default threshold for SFace

            for name, known_embedding in known_embeddings.items():
                similarity = cosine_similarity(face_embedding, known_embedding)
                if similarity > max_similarity and similarity > threshold:
                    max_similarity = similarity
                    recognized_name = name
        
            # Track movement
            # print(f"Calling track_movement for: {recognized_name} at ({center_x}, {center_y})")
            track_movement(recognized_name, center_x, center_y, get_db)


            # Draw bounding box & name
            color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, recognized_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame  # Return modified frame with recognized faces
