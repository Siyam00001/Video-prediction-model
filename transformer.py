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

RESIZE_DIM = (64, 64)
INPUT_FRAMES = 10
PRED_FRAMES = 5

def preprocess_video_for_inference(video_path, resize_dim):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, resize_dim)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError("No frames were read from the video")
    return np.array(frames)

def evaluate_predictions(true_frames, pred_frames):
    if true_frames.shape != pred_frames.shape:
        raise ValueError("True and predicted frames must have the same shape")
    mse = np.mean((true_frames - pred_frames) ** 2)
    ssim_scores = []
    for t, p in zip(true_frames, pred_frames):
        score = structural_similarity(t, p, data_range=255)
        ssim_scores.append(score)
    return {'MSE': mse, 'SSIM': np.mean(ssim_scores)}

def evaluate_and_display_metrics(true_frames, pred_frames):
    metrics = evaluate_predictions(true_frames, pred_frames)
    st.subheader("Evaluation Metrics")
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

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        
    def forward(self, x):
        batch_size, seq_len, n_patches, _ = x.shape
        x = x.reshape(batch_size * n_patches, seq_len, self.dim)
        x = x.transpose(0, 1)
        attn_output, _ = self.att(x, x, x)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(batch_size, n_patches, seq_len, self.dim)
        attn_output = attn_output.transpose(1, 2)
        return attn_output

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class VideoPatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, emb_size=128):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        return self.projection(x)

class VideoTransformer(nn.Module):
    def __init__(self, in_channels=1, img_size=64, patch_size=8, emb_dim=128,
                 n_layers=6, n_heads=8, dropout=0.1, input_frames=10, 
                 pred_frames=5):
        super().__init__()
        self.patch_size = patch_size
        self.input_frames = input_frames
        self.pred_frames = pred_frames
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = VideoPatchEmbedding(in_channels, patch_size, emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_frames, self.num_patches, emb_dim))
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=n_heads, dropout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim * 4, dropout=dropout)))
            ))
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size * in_channels),
            Rearrange('b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)',
                     h=img_size//patch_size, w=img_size//patch_size,
                     p1=patch_size, p2=patch_size)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embedding
        for layer in self.layers:
            x = layer(x)
        pred_frames = []
        curr_input = x
        for _ in range(self.pred_frames):
            last_state = curr_input[:, -1:]
            next_frame = self.decoder(last_state)
            pred_frames.append(next_frame)
            next_embedded = self.patch_embed(next_frame)
            curr_input = torch.cat([curr_input[:, 1:], next_embedded], dim=1)
        return torch.cat(pred_frames, dim=1)

def prepare_input_tensor(frames):
    if len(frames) < INPUT_FRAMES:
        raise ValueError(f"Not enough frames. Expected {INPUT_FRAMES}, got {len(frames)}")
    frames = torch.FloatTensor(frames[:INPUT_FRAMES])
    frames = frames.unsqueeze(0).unsqueeze(2) / 255.0
    return frames

def display_frames_grid(frames):
    cols = st.columns(len(frames))
    for idx, col in enumerate(cols):
        col.image(frames[idx], caption=f"Frame {idx+1}", use_column_width=True)


def display_video_frames(input_frames, pred_frames):
    all_frames = np.concatenate([input_frames, pred_frames])
    
    
    frame_placeholder = st.empty()
    
    # continous frams display
    while True:
        for frame in all_frames:
            frame_placeholder.image(frame, caption="Video Preview", use_column_width=True)
            time.sleep(0.2)  # Control frame rate

def create_app():
    st.title("Video Frame Prediction")
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
                    frames = preprocess_video_for_inference(video_path, RESIZE_DIM)
                
                input_tensor = prepare_input_tensor(frames)
                
                start_time = time.time()
                with torch.no_grad():
                    predictions = model(input_tensor.to(device))
                inference_time = time.time() - start_time
                
                st.subheader("Results")
                st.write(f"Inference time: {inference_time:.2f} seconds")
                
                pred_frames = predictions.cpu().numpy().squeeze() * 255
                pred_frames = pred_frames.astype(np.uint8)
                
                evaluate_and_display_metrics(
                    frames[INPUT_FRAMES:INPUT_FRAMES+PRED_FRAMES],
                    pred_frames
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
    try:
        model = torch.load('VisualTransformer.pth', map_location=device)
        model = model.to(device)
        model.eval()
        create_app()
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'VisualTransformer.pth' exists.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")