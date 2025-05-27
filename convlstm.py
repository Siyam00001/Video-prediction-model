# convlstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Constants
FRAME_SIZE = (64, 64)
NUM_FRAMES = 10
NUM_CLASSES = 6

def preprocess_video(video_path, num_frames=NUM_FRAMES):
    """Preprocess video for ConvLSTM input"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // (num_frames + 2), 1)
    
    try:
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize and convert to grayscale
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            frame_count += 1
            
            # Skip frames according to frame_step
            for _ in range(frame_step - 1):
                cap.read()
    finally:
        cap.release()
    
    # Convert to numpy array
    frames = np.array(frames)
    
    # If we don't have enough frames, pad with zeros
    if len(frames) < num_frames:
        padding = ((0, num_frames - len(frames)), (0, 0), (0, 0))
        frames = np.pad(frames, padding, mode='constant')
        
    # Ensure we have exactly num_frames
    frames = frames[:num_frames]
    
    # Print shape for debugging
    print(f"Preprocessed frames shape: {frames.shape}")
    return frames

def prepare_input(frames):
    """Prepare input frames for prediction"""
    print(f"Input frames shape before preparation: {frames.shape}")
    
    # Normalize to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # Reshape to (batch_size, sequence_length, height, width, channels)
    frames = frames.reshape(1, NUM_FRAMES, FRAME_SIZE[0], FRAME_SIZE[1], 1)
    
    print(f"Input frames shape after preparation: {frames.shape}")
    return frames

def load_convlstm_model(model_path):
    """Load the saved ConvLSTM model"""
    try:
        with tf.device('/CPU:0'):
            model = load_model(model_path)
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            return model
    except Exception as e:
        raise Exception(f"Error loading ConvLSTM model: {str(e)}")

def generate_predictions(model, input_frames):
    """Generate predictions using the model"""
    try:
        with tf.device('/CPU:0'):
            predictions = model.predict(input_frames, verbose=0)
        return predictions
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")