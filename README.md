# Video prediction model
Sure! Here's a **GitHub README** for your project:

---

# **Predicting and Generating Video Sequences Using Deep Learning**

## **Overview**
This project focuses on **video prediction**, generating future frames from short input sequences using the **UCF101 dataset**. It explores three deep learning architectures:
- **ConvLSTM** (Convolutional LSTM)
- **PredRNN** (Predictive Recurrent Neural Network)
- **Transformer-based Model**

By learning motion patterns, this model supports applications in **video synthesis, animation, and scene prediction**. A **user-friendly interface** allows visualization of generated sequences.

## **Features**
✔️ Predict future video frames  
✔️ Supports ConvLSTM, PredRNN, and Transformer-based architectures  
✔️ Evaluates results with **MSE**, **SSIM**, and **PSNR**  
✔️ Interactive **Streamlit interface** for visualizing predictions  

## **Architectures**
### 🔹 **ConvLSTM**
- Captures **spatial and temporal dependencies** using stacked ConvLSTM layers.
- Uses **categorical cross-entropy loss** and **Adam optimizer**.
- Performance Metrics:
  - **MSE**: 105.14  
  - **SSIM**: 0.0114  
  - **PSNR**: 7.49 dB  

### 🔹 **PredRNN**
- Improves **temporal modeling** via **spatiotemporal memory blocks**.
- Training with **categorical cross-entropy loss** and **Adam optimizer**.
- Performance Metrics:
  - **MSE**: 105.70  
  - **SSIM**: 0.0084  
  - **PSNR**: 10.34 dB  

### 🔹 **Transformer-based Model**
- Uses **self-attention mechanisms** for enhanced **long-term dependencies**.
- **Patch-based embeddings** improve prediction accuracy.
- Performance Metrics:
  - **MSE**: **0.0058** ✅ (Lowest)  
  - **SSIM**: **0.7796** ✅ (Highest)  
  - **PSNR**: **22.35 dB** ✅ (Best Quality)  

## **Installation**
1️⃣ Clone this repository:  
```bash
git clone https://github.com/your-username/video-prediction.git
```
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```
3️⃣ Run the Streamlit interface:  
```bash
streamlit run app.py
```

## **Usage**
1. **Upload a video** or choose one from the **UCF101 dataset**.
2. **Select a model** (ConvLSTM, PredRNN, Transformer).
3. **Generate predictions** and visualize input vs. predicted frames.
4. **Evaluate frame quality** using MSE, SSIM, and PSNR.

## **Dataset**
- **UCF101 Dataset**  
- Contains **101 human activity classes**  
- Frames are resized to **64×64 grayscale**  

## **Results**
| Model        | MSE    | SSIM  | PSNR (dB) |
|-------------|--------|-------|-----------|
| ConvLSTM    | 105.14 | 0.0114 | 7.49      |
| PredRNN     | 105.70 | 0.0084 | 10.34     |
| Transformer | **0.0058** | **0.7796** | **22.35** |

## **Challenges**
🔸 **ConvLSTM:** High MSE, struggles with long-term dependencies.  
🔸 **PredRNN:** Complex temporal modeling, requires **high computational power**.  
🔸 **Transformer:** Best performance but **high memory demands**.

## **Future Work**
🚀 **Optimize Transformer model** for efficiency  
🚀 **Explore attention-based frame refinement**  
🚀 **Deploy as a web application**  

## **Contributors**
👨‍💻 **Siyam Haider** – FAST, NUCES Islamabad, Pakistan  

## **License**
This project is licensed under the **MIT License**.

---

Let me know if you'd like any modifications! 🚀😃

