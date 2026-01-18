---
title: AI NIDS Student Project
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# 🛡️ AI-Based Network Intrusion Detection System (Student Project)

This project demonstrates how to use **Machine Learning (Random Forest)** and **Generative AI (Grok)** to detect and explain network attacks (specifically DDoS).

## ✨ Features

- **Random Forest Classifier**: Trained on the CIC-IDS2017 dataset to classify network traffic as BENIGN or DDoS
- **Model Persistence**: Trained models are saved to disk and automatically loaded on app restart
- **Feature Importance Visualization**: Bar chart showing which packet fields matter most for detection
- **Confusion Matrix**: Visual breakdown of model accuracy and prediction distribution
- **AI-Powered Explanations**: Uses Groq (Llama 3.3 70B) to explain why a packet was flagged
- **Remediation Suggestions**: Optional AI-generated security recommendations for detected threats

## 🚀 How to Use

1. **Enter API Key:** Paste your Grok API key in the sidebar (optional, for AI explanations).
2. **Train Model:** Click the "Train AI Model" button. The system loads the `Friday-WorkingHours...` dataset automatically.
3. **Explore Performance:** Check the "Model Performance" tab to see feature importance and confusion matrix.
4. **Simulate:** Click "Capture Random Packet" to pick a real network packet from the test set.
5. **Analyze:** See if the model flags it as **BENIGN** or **DDoS**, and ask Groq to explain why.

## 📂 Files

| File | Description |
|------|-------------|
| `app.py` | The main Streamlit application |
| `requirements.txt` | Python dependencies |
| `trained_model.joblib` | Saved model (created after first training) |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | CIC-IDS2017 dataset subset |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn (Random Forest)
- **AI**: Groq API (Llama 3.3 70B Versatile)
- **Data**: pandas, numpy

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 15
- **Features Used**:
  - Flow Duration
  - Total Fwd/Backward Packets
  - Total Length of Fwd Packets
  - Fwd Packet Length Max
  - Flow IAT Mean/Std
  - Flow Packets/s

## 🎓 About

Created for a university cybersecurity project to demonstrate the integration of traditional ML and LLMs in security operations.# AI-Based-Network-Intrusion-Detection-System
