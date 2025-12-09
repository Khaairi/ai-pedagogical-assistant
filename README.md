# Student Engagement Analytics with Computer Vision & GenAI
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![LangChain](https://img.shields.io/badge/LangChain-GenAI-green)
![Gemini](https://img.shields.io/badge/Google-Gemini%20Flash-8E75B2)

> **An AI-powered application designed to analyze student facial expressions in classroom videos and generate automated pedagogical insights using Large Language Models.**

Model development / training: https://github.com/Khaairi/Facial_Expression_Recognition_Hybrid_ViT_Mobilenetv3.git

## Overview
This project combines **Deep Learning (Computer Vision)** for emotion recognition with **Generative AI (LLM)** to provide actionable feedback for educators.

The system processes classroom videos to detect student emotions over time (e.g., Happy, Sad, Angry) and uses **Google Gemini** to act as a "Virtual Pedagogical Consultant," interpreting the data and suggesting teaching improvements.

## Key Features
* **Hybrid Computer Vision Model:** Utilizes a custom **MobileNetV3 + Vision Transformer (ViT)** architecture for lightweight yet accurate facial expression recognition.
* **Face Detection:** Implements **RetinaFace** for robust face detection in classroom environments.
* **Real-time Analytics:** Visualizes emotion frequency trends per second using dynamic charts.
* **AI Pedagogical Insights:** Integrates **Google Gemini 1.5 Flash** (via LangChain/Direct Prompting) to read the analysis data and generate a narrative report with specific solutions for the teacher.
* **Data Export:** Allows users to download the raw analysis data as CSV.

## Tech Stack
* **Framework:** Streamlit
* **Deep Learning:** PyTorch, Torchvision
* **Model Architecture:** Hybrid MobileNetV3-ViT (Custom trained)
* **Face Detection:** RetinaFace
* **Generative AI:** Google Gemini API (via `langchain-google-genai`)
* **Data Processing:** Pandas, NumPy, OpenCV
