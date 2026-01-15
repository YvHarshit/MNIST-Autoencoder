# MNIST Autoencoder using TensorFlow

## 📌 Project Overview
This project implements a **Deep Autoencoder** using **TensorFlow and Keras** to learn **compressed representations (features)** of handwritten digit images from the **MNIST dataset**.

Unlike classification models, this project focuses on **unsupervised learning**, where the model learns patterns and structures in the data **without using labels**. The trained autoencoder reconstructs the original input images from a low-dimensional latent space.

---

## ❓ What is an Autoencoder?
An **autoencoder** is a type of neural network designed to:
- Compress input data into a smaller representation (encoding)
- Reconstruct the original data from this compressed form (decoding)

It consists of:
- **Encoder** → Feature extraction & compression
- **Latent Space** → Compact representation of data
- **Decoder** → Reconstruction of the original input

---

## 🎯 Why This Project?
The goal of this project is to:
- Understand **unsupervised learning**
- Learn **feature extraction and dimensionality reduction**
- Demonstrate **image reconstruction**
- Explore how neural networks learn meaningful representations without labels

This project is useful for understanding the foundation of:
- Image compression
- Denoising
- Anomaly detection
- Representation learning

---

## 🧠 How the Project Works

### 1️⃣ Dataset
- **MNIST Handwritten Digits**
- Image size: `28 × 28`
- Grayscale images
- Labels are intentionally ignored

### 2️⃣ Model Architecture

**Encoder:**
28×28 → 784 → 128 → 64 → 32

**Decoder:**
32 → 64 → 128 → 784 → 28×28

The latent space size is **32**
- Activation functions: ReLU (hidden layers), Sigmoid (output)
- Loss function: Binary Cross-Entropy
- Optimizer: Adam

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Run the Script
python mnist_autoencoder.py
