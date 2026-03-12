import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pygame import mixer

# 1. Define the exact same PyTorch CNN Architecture used in training
class EyeClassifier(nn.Module):
    def __init__(self):
        super(EyeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict_eye_state(eye_img, model, device):
    """Preprocess the cropped eye image and run it through the PyTorch model."""
    eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24))
    eye = eye / 255.0  # Normalize
    eye = eye.reshape(1, 1, 24, 24)  # Reshape to (Batch, Channels, Height, Width)
    
    eye_tensor = torch.tensor(eye, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(eye_tensor)
        prediction = torch.argmax(output, dim=1).item()
        
    return prediction  # 0 for Closed, 1 for Open

def main():
    # 2. Setup the Environment and Load Assets
    mixer.init()
    try:
        sound = mixer.Sound('alarm.wav')
    except FileNotFoundError:
        print("Warning: 'alarm.wav' not found. Alarm will not play.")
        sound = None

    # Load Haar Cascades using OpenCV's built-in paths (No local folder needed)
    cascade_path = cv2.data.haarcascades
    leye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_lefteye_2splits.xml'))
    reye_cascade = cv2.CascadeClassifier(os.path.join(cascade_path, 'haarcascade_righteye_2splits.xml'))

    # 3. Initialize PyTorch Model and Load Weights
    # Optimized for Mac CPU or Apple Silicon MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running inference on: {device}")
    
    model = EyeClassifier().to(device)
    try:
        model.load_state_dict(torch.load('models/model.pth', map_location=device, weights_only=True))
        print("Successfully loaded 'models/model.pth'")
    except FileNotFoundError:
        print("Error: 'models/model.pth' not found. Please run the training script first.")
        return

    model.eval()  # Set model to evaluation mode

    # 4. Main Detection Loop
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score = 0
    thicc = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break
            
        # Mirror frame to act like a natural mirror
        frame = cv2.flip(frame, 1)    
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ONLY eyes (Removed face detection logic)
        # Added minNeighbors=5 to reduce false positives
        left_eye = leye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        right_eye = reye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        
        # Draw bottom black rectangle for UI text
        cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
            
        lpred, rpred = 99, 99  # Default invalid states
        
        # Process Right Eye
        for (x, y, w, h) in right_eye:
            # Draw Blue bounding box for Right Eye
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Right", (x, y-5), font, 0.7, (255, 0, 0), 1)
            
            r_eye_img = frame[y:y+h, x:x+w]
            rpred = predict_eye_state(r_eye_img, model, device)
            break  # Only process the first detected right eye
            
        # Process Left Eye
        for (x, y, w, h) in left_eye:
            # Draw Green bounding box for Left Eye
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Left", (x, y-5), font, 0.7, (0, 255, 0), 1)
            
            l_eye_img = frame[y:y+h, x:x+w]
            lpred = predict_eye_state(l_eye_img, model, device)
            break  # Only process the first detected left eye
            
        # 5. Logic for Scoring and Alarming
        # Both eyes must be closed to increase the drowsiness score
        if lpred == 0 and rpred == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
        if score < 0:
            score = 0
            
        cv2.putText(frame, f'Score:{score}', (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Trigger alarm if score exceeds threshold
        if score > 15:
            cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame) # Save proof image
            if sound:
                try:
                    sound.play()
                except Exception:
                    pass
                
            # Create a pulsing red border effect
            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
            if thicc < 2:
                thicc = 2
                
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            
        cv2.imshow('Drowsiness Detection System', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()