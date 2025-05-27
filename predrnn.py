# predrnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import cv2
import os

# Constants
FRAME_SIZE = (64, 64)
NUM_FRAMES = 10
NUM_CLASSES = 6

def build_predrnn(input_shape, num_classes):
    model = models.Sequential([
        # Spatial-temporal convolutions
        layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', 
                     padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', 
                     padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # ConvLSTM layers
        layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu', 
                         return_sequences=True),
        layers.BatchNormalization(),
        
        layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='relu', 
                         return_sequences=False),
        layers.BatchNormalization(),

        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def preprocess_video(video_path, num_frames=NUM_FRAMES):
    """Preprocess video to get exactly num_frames frames"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample evenly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    try:
        frame_idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in indices:
                frame = cv2.resize(frame, FRAME_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
                
            frame_idx += 1
            
    finally:
        cap.release()
    
    frames = np.array(frames)
    
    # Handle cases where we couldn't get enough frames
    if len(frames) < num_frames:
        # Pad with copies of the last frame
        last_frame = frames[-1] if len(frames) > 0 else np.zeros(FRAME_SIZE)
        padding = [(0, num_frames - len(frames)), (0, 0), (0, 0)]
        frames = np.pad(frames, padding, mode='constant', constant_values=0)
    
    # Ensure exact number of frames
    frames = frames[:num_frames]
    
    print(f"Preprocessed frames shape: {frames.shape}")
    return frames

def prepare_input(frames):
    """Prepare input frames for prediction"""
    # Ensure we have exactly NUM_FRAMES frames
    if frames.shape[0] != NUM_FRAMES:
        raise ValueError(f"Expected {NUM_FRAMES} frames, got {frames.shape[0]}")
        
    # Normalize and reshape
    frames = frames.astype(np.float32) / 255.0
    frames = frames.reshape(1, NUM_FRAMES, FRAME_SIZE[0], FRAME_SIZE[1], 1)
    
    print(f"Prepared input shape: {frames.shape}")
    return frames



def load_predrnn_model(model_path):
    """Load the saved PredRNN model"""
    try:
        with tf.device('/CPU:0'):
            model = load_model(model_path)
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            return model
    except Exception as e:
        raise Exception(f"Error loading PredRNN model: {str(e)}")

def generate_predictions(model, input_frames):
    """Generate predictions using the model"""
    try:
        with tf.device('/CPU:0'):
            predictions = model.predict(input_frames, verbose=0)
        return predictions
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")