import os
import cv2
import numpy as np
import glob

def resize_image(image, target_size=(160, 160)):
    """Resizes an image while maintaining aspect ratio and pads if needed."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)  # Scale to fit within 320x320
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize while keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank (black) 320x320 image and place the resized image in the center
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

# Function to train image
def train_image(image_url, user_name, user_id):
    DETECTION_MODEL = "models/face_detection_yunet_2023mar.onnx"
    RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"
    
    os.makedirs("detected_faces", exist_ok=True)
    os.makedirs("face_embeddings", exist_ok=True)
    
    detector = cv2.FaceDetectorYN.create(DETECTION_MODEL, "", (320, 320))
    recognizer = cv2.FaceRecognizerSF.create(RECOGNITION_MODEL, "")
    
    image = cv2.imread(image_url)
    if image is None:
        print('No image found')
        return False, "Image not found"
    
    # Resize image
    image = resize_image(image, (320, 320))
    
    height, width, _ = image.shape
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    
    if faces is None:
        print('No face detected')
        return False, "No faces detected"
    
    num_faces = faces.shape[0]
    if num_faces > 1:
        print('Multiple faces detected')
        return False, "Multiple faces detected. Please provide an image with only one face."
    
    # --- Determine next face number ---
    existing_faces = glob.glob(f"detected_faces/{user_id}_{user_name}_face_*.jpg")
    next_face_num = len(existing_faces) + 1
    
    # --- Save face image and embedding ---
    face = faces[0]
    x, y, w, h = map(int, face[:4])
    cropped_face = image[y:y+h, x:x+w]
    
    face_path = f"detected_faces/{user_id}_{user_name}_face_{next_face_num}.jpg"
    embedding_path = f"face_embeddings/{user_id}_{user_name}_face_{next_face_num}.npy"
    
    cv2.imwrite(face_path, cropped_face)
    
    aligned_face = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned_face)
    np.save(embedding_path, embedding)
    
    print(f"Training successful, saved as {face_path}")
    return True, f"Training successful, saved as {face_path}"
# uncomment the line below to see the images get trained
# train_image("static/images/shehzad_collage.jpg", "246")
