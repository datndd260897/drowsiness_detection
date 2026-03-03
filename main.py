import cv2
import dlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from imutils import face_utils

# Re-define model architecture to load the weights
class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load PyTorch Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EyeClassifier()
model.load_state_dict(torch.load('eye_model.pth', map_location=device))
model.to(device)
model.eval()

# Image transforms for inference
transform = transforms.Compose([
    transforms.Resize((24, 24)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get indices for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def predict_eye_state(eye_img):
    """Takes a cropped cv2 eye image, runs it through PyTorch, returns Open(1) or Closed(0)"""
    # Convert BGR (OpenCV) to Grayscale PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY))
    tensor_img = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_img).item()
    return 1 if output > 0.5 else 0 # 1 is Open, 0 is Closed

# Setup MacBook camera (index 0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

CONSEC_FRAMES = 15 # Trigger alarm if eyes closed for 15 frames
COUNTER = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mirror the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Draw bounding boxes around eyes on the main frame
        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        # Get bounding boxes to crop the eyes out for the model
        (xL, yL, wL, hL) = cv2.boundingRect(np.array([leftEye]))
        (xR, yR, wR, hR) = cv2.boundingRect(np.array([rightEye]))
        
        # Add a little padding to the crops
        pad = 5
        left_eye_crop = frame[max(0, yL-pad):yL+hL+pad, max(0, xL-pad):xL+wL+pad]
        right_eye_crop = frame[max(0, yR-pad):yR+hR+pad, max(0, xR-pad):xR+wR+pad]

        # Display the cropped eye frames (As requested)
        if left_eye_crop.size > 0 and right_eye_crop.size > 0:
            # Resize them to make them easier to see on screen
            show_l = cv2.resize(left_eye_crop, (150, 100))
            show_r = cv2.resize(right_eye_crop, (150, 100))
            # Stack horizontally and show
            eyes_combined = np.hstack([show_l, show_r])
            cv2.imshow("Extracted Eyes", eyes_combined)

            # Predict State
            pred_left = predict_eye_state(left_eye_crop)
            pred_right = predict_eye_state(right_eye_crop)

            # If BOTH eyes are closed
            if pred_left == 0 and pred_right == 0:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            else:
                COUNTER = 0

        # Display counters/stats
        state_text = "Closed" if (COUNTER > 0) else "Open"
        cv2.putText(frame, f"Eyes: {state_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("MacBook Drowsiness Monitor", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()