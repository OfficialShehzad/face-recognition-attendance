import cv2
from mtcnn import MTCNN  # Import MTCNN for face detection

# Initialize the MTCNN face detector
detector = MTCNN()

def detect_faces(frame):
    """Function to detect faces in the frame using MTCNN."""
    
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to a lower resolution (e.g., 640x480)
    faces = detector.detect_faces(frame_resized)
    
    # Iterate over the faces detected in the frame
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        
        confidence = face['confidence']  # The confidence score for the face detection

        # Draw bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw facial landmarks (keypoints) such as eyes, nose, and mouth
        for key, point in face['keypoints'].items():
            cv2.circle(frame, point, 2, (0, 0, 255), 2)  # Draw keypoints (eyes, nose, etc.)

        # Optionally display the confidence score near the face
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame
