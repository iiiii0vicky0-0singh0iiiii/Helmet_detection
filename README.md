# 🪖 Helmet Detection System using YOLOv8

🚀 Real-time Helmet Detection System built using **YOLOv8** and deployed with a **Streamlit dashboard** to detect riders with and without helmets.

---

## 🌐 Live Demo
👉 https://helmetdetection-up.streamlit.app/

---

## 📌 Project Overview

This project uses **Computer Vision and Deep Learning** to detect whether a person is wearing a helmet or not.

It is designed for:
- 🚦 Traffic monitoring systems  
- 🚔 Safety enforcement  
- 🏍️ Rider compliance detection  

---

## ⚙️ Features

- ✅ Helmet vs No Helmet detection  
- ✅ Custom trained YOLOv8 model  
- ✅ Image upload detection  
- ✅ Bounding box visualization  
- ✅ Detection count (Helmet / No Helmet)  
- ✅ Download result image  
- ✅ Streamlit web dashboard  

---

## 🧠 Tech Stack

- **Language:** Python  
- **Model:** YOLOv8 (Ultralytics)  
- **Libraries:** OpenCV, NumPy, Pillow  
- **Framework:** Streamlit  
- **Tools:** Git, GitHub  

---

## 📂 Project Structure
<img width="319" height="261" alt="image" src="https://github.com/user-attachments/assets/8cdf722e-58c1-4d5f-be67-ade48d626e50" />

---
## 🏋️ Model Training

- Dataset annotated in **Pascal VOC format**
- Converted to **YOLO format**
- Trained using:
```bash

yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

