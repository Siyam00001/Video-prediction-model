import streamlit as st
import time
from pathlib import Path 
import torch
import cv2
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import Dataset
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from transformer import PreNorm, FeedForward, Attention, VideoTransformer, VideoPatchEmbedding, ResidualAdd, RESIZE_DIM
from convlstm import (load_convlstm_model, preprocess_video as preprocess_convlstm_video, 
                     prepare_input as prepare_convlstm_input)
import tensorflow as tf

# Constants
INPUT_FRAMES = 10
PRED_FRAMES = 5
FRAME_SIZE = (64, 64)
NUM_FRAMES = 10 

# Preprocessing function for Transformer
def preprocess_video_for_transformer(video_path, resize_dim):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    frames = []
    try:
        while True:
            ret, frame = cap.read() 
            if not ret: 
                break
            frame = cv2.resize(frame, resize_dim)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError("No frames were read from the video")
    return np.array(frames)

def prepare_transformer_input(frames):
    if len(frames) < INPUT_FRAMES:
        raise ValueError(f"Not enough frames. Expected {INPUT_FRAMES}, got {len(frames)}")
    frames = torch.FloatTensor(frames[:INPUT_FRAMES])
    frames = frames.unsqueeze(0).unsqueeze(2) / 255.0
    return frames

# Evaluation functions
def evaluate_predictions(true_frames, pred_frames):
    if true_frames.shape != pred_frames.shape:
        raise ValueError("True and predicted frames must have the same shape")
    mse = np.mean((true_frames - pred_frames) ** 2)
    ssim_scores = []
    for t, p in zip(true_frames, pred_frames):
        score = structural_similarity(t, p, data_range=255)
        ssim_scores.append(score)
    return {'MSE': mse, 'SSIM': np.mean(ssim_scores)}

def evaluate_and_display_metrics(true_frames, pred_frames, model_type):
    metrics = evaluate_predictions(true_frames, pred_frames)
    st.subheader(f"{model_type} Evaluation Metrics")
    col1, col2 = st.columns(2)
    
    col1.markdown(f"""
        <div style='padding: 20px; border-radius: 5px; background-color: #f0f2f6;'>
            <h3 style='text-align: center; color: #1f77b4;'>MSE</h3>
            <h2 style='text-align: center;'>{metrics['MSE']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    col2.markdown(f"""
        <div style='padding: 20px; border-radius: 5px; background-color: #f0f2f6;'>
            <h3 style='text-align: center; color: #1f77b4;'>SSIM</h3>
            <h2 style='text-align: center;'>{metrics['SSIM']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

# Display functions
def display_frames_grid(frames):
    cols = st.columns(len(frames))
    for idx, col in enumerate(cols):
        col.image(frames[idx], caption=f"Frame {idx+1}", use_column_width=True)

def display_video_frames(input_frames, pred_frames):
    all_frames = np.concatenate([input_frames, pred_frames])
    frame_placeholder = st.empty()
    while True:
        for frame in all_frames:
            frame_placeholder.image(frame, caption="Video Preview", use_column_width=True)
            time.sleep(0.2)

def load_model(model_type, device):
    try:
        if model_type == "Transformer":
            model = torch.load('VisualTransformer.pth', map_location=device)
            return ("torch", model.to(device).eval())
        elif model_type == "ConvLSTM":
            model = load_convlstm_model("optimized_CL_model_scratch123.keras")
            return ("keras", model)
        elif model_type == "PredRNN":
            model = torch.load('PredRNN.pth', map_location=device)
            return ("torch", model.to(device).eval())
    except FileNotFoundError:
        st.error(f"{model_type} model file not found")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None



def prepare_convlstm_input(frames):
    """Prepare input for ConvLSTM model"""
    # verification of number of frames
    if len(frames) > NUM_FRAMES:
        frames = frames[:NUM_FRAMES]
    elif len(frames) < NUM_FRAMES:
        # Pad with zeros in case of fewer frames
        padding = ((0, NUM_FRAMES - len(frames)), (0, 0), (0, 0))
        frames = np.pad(frames, padding, mode='constant')

    # Normalize and reshape
    frames = frames.astype(np.float32) / 255.0
    frames = frames.reshape(1, NUM_FRAMES, FRAME_SIZE[0], FRAME_SIZE[1], 1)
    return frames



def create_app():
    st.title("Video Frame Prediction")
    
    model_type = st.selectbox("Select Model", ["Transformer", "ConvLSTM", "PredRNN"])
    
    uploaded_file = st.file_uploader("Upload a video clip", type=['mp4', 'avi'])
    
    dataset_path = Path("preprocessed_dataset/test")
    if not dataset_path.exists():
        st.error("Dataset directory not found")
        return
        
    action_classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    selected_action = st.selectbox("Select action:", action_classes)
    
    if selected_action:
        action_path = dataset_path / selected_action
        video_files = sorted([d.name for d in action_path.iterdir() if d.is_dir()])
        selected_file = st.selectbox("Select video:", video_files)
    
    if st.button("Generate Prediction"):
        try:
            model_info = load_model(model_type, device)
            if model_info is None:
                return
            
            model_framework, model = model_info
                
            if uploaded_file:
                temp_path = f"temp_{uploaded_file.name}"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    video_path = temp_path
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                video_path = str(dataset_path/selected_action/selected_file/'frames.npy')
                
            with st.spinner("Processing video..."):
                if video_path.endswith('.npy'):
                    frames = np.load(video_path)
                else:
                    if model_framework == "keras":
                        frames = preprocess_convlstm_video(video_path)
                    else:
                        frames = preprocess_video_for_transformer(video_path, RESIZE_DIM)
                
                start_time = time.time()
                
                if model_framework == "keras":
                    try:
                        with tf.device('/CPU:0'):
                            print(f"Original frames shape: {frames.shape}")
                            # Prepare input for ConvLSTM
                            input_tensor = prepare_convlstm_input(frames)
                            print(f"Input tensor shape: {input_tensor.shape}")
                            
                            # Get predictions
                            predictions = model.predict(input_tensor, verbose=0)
                            print(f"Raw predictions shape: {predictions.shape}")
                            
                            # Display classification results
                            st.subheader("Action Predictions")
                            action_classes = ["Basketball", "Typing", "PlayingGuitar", "PullUps", "SoccerJuggling", "Rowing"]
                            for i, prob in enumerate(predictions[0]):
                                st.write(f"{action_classes[i]}: {prob:.2%}")
                            
                            # For visualization, show the input video frames
                            st.subheader("Input Video Frames")
                            display_frames_grid(frames[:INPUT_FRAMES])
                            
                            # Show video animation of the input frames
                            st.subheader("Video Animation")
                            frame_placeholder = st.empty()
                            while True:
                                for frame in frames[:INPUT_FRAMES]:
                                    frame = frame.astype(np.uint8)
                                    frame_placeholder.image(frame, caption="Video Preview", use_column_width=True)
                                    time.sleep(0.2)
                                    
                    except Exception as e:
                        st.error(f"Error during ConvLSTM prediction: {str(e)}")
                        st.error(f"Debug info: frames shape = {frames.shape}")
                        return
                else:
                    input_tensor = prepare_transformer_input(frames)
                    with torch.no_grad():
                        predictions = model(input_tensor.to(device))
                    pred_frames = predictions.cpu().numpy().squeeze() * 255
                
                inference_time = time.time() - start_time
                
                st.subheader("Results")
                st.write(f"Inference time: {inference_time:.2f} seconds")
                
                if model_framework == "keras":
                    # Display classification results for ConvLSTM
                    st.subheader("Action Predictions")
                    action_classes = ["Basketball", "Typing", "PlayingGuitar", "PullUps", "SoccerJuggling", "Rowing"]
                    for i, prob in enumerate(predictions[0]):
                        st.write(f"{action_classes[i]}: {prob:.2%}")
                else:
                    # Display frame predictions for Transformer
                    pred_frames = pred_frames.astype(np.uint8)
                    evaluate_and_display_metrics(
                        frames[INPUT_FRAMES:INPUT_FRAMES+PRED_FRAMES],
                        pred_frames,
                        model_type
                    )
                    
                    st.subheader("Input Frames")
                    display_frames_grid(frames[:INPUT_FRAMES])
                    
                    st.subheader("Predicted Frames") 
                    display_frames_grid(pred_frames)
                    
                    st.subheader("Video Animation")
                    display_video_frames(frames[:INPUT_FRAMES], pred_frames)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    create_app()