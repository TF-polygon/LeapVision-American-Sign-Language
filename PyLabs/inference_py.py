#!/usr/bin/env python3
"""
ASL Recognition Inference Script
Real-time ASL recognition using trained PBN model
"""

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from train import PatchBasedNetwork
import mediapipe as mp

class ASLPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = PatchBasedNetwork(num_classes=26)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # ASL alphabet mapping
        self.classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def predict_image(self, image):
        """Predict ASL letter from image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        return self.classes[predicted.item()], confidence.item()
    
    def extract_hand_roi(self, frame):
        """Extract hand region using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box
                h, w, _ = frame.shape
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
                
                # Add padding
                padding = 50
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Extract ROI
                hand_roi = frame[y_min:y_max, x_min:x_max]
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                return hand_roi, (x_min, y_min, x_max, y_max)
        
        return None, None
    
    def run_webcam_inference(self):
        """Run real-time ASL recognition from webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting ASL Recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Extract hand ROI
            hand_roi, bbox = self.extract_hand_roi(frame)
            
            if hand_roi is not None and hand_roi.size > 0:
                try:
                    # Predict ASL letter
                    predicted_letter, confidence = self.predict_image(hand_roi)
                    
                    # Draw bounding box and prediction
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Display prediction
                    text = f"{predicted_letter}: {confidence:.2f}"
                    cv2.putText(frame, text, (x_min, y_min-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Show frame
            cv2.imshow('ASL Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize predictor
    predictor = ASLPredictor('pbn_asl_model.pth')
    
    # Run webcam inference
    predictor.run_webcam_inference()

if __name__ == "__main__":
    main()