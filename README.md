<div align="center">

# 🪖 Helmet Detection System

### Real-time rider safety detection powered by **YOLOv8** and deployed via **Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9500?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

**[🚀 Live Demo](https://helmetdetection-up.streamlit.app/)** · **[Report Bug](https://github.com/yourusername/helmet-detection-system/issues)** · **[Request Feature](https://github.com/yourusername/helmet-detection-system/issues)**

<br>

![Demo Banner](https://img.shields.io/badge/STATUS-LIVE%20%26%20DEPLOYED-brightgreen?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Use Cases](#-use-cases)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Detection Pipeline](#-detection-pipeline)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

A production-ready **Computer Vision system** built on **YOLOv8** to automatically detect whether motorcycle and bicycle riders are wearing helmets.

Designed for:
- 🚦 Traffic safety enforcement and smart city monitoring
- 🚔 Law enforcement assistance with AI-assisted violation detection
- 📈 Rider compliance analytics and road safety research

| Model | Classes | Framework | Deployment |
|-------|---------|-----------|------------|
| YOLOv8 (Custom Trained) | Helmet / No Helmet | Streamlit | Cloud (Streamlit Cloud) |

---

## 🌐 Live Demo

👉 **[https://helmetdetection-up.streamlit.app/](https://helmetdetection-up.streamlit.app/)**

> Upload any image of a rider and get instant helmet detection results with bounding boxes and confidence scores.

---

## ⚙️ Features

| Feature | Description |
|---------|-------------|
| ✅ **Dual-Class Detection** | Detects both *Helmet* and *No Helmet* riders in a single pass |
| ✅ **Image Upload** | Upload JPG / PNG / WEBP via the Streamlit UI |
| ✅ **Bounding Box Visualization** | Color-coded annotations with confidence scores |
| ✅ **Detection Count** | Real-time summary — total Helmet vs No Helmet counts |
| ✅ **Download Results** | Export the annotated output image with one click |
| ✅ **Custom Trained Model** | YOLOv8 fine-tuned on a domain-specific helmet dataset |
| ✅ **Web Dashboard** | Clean, responsive Streamlit interface — no local setup needed |

---

## 🎯 Use Cases

```
🚦 Traffic Monitoring     →  Integrate with CCTV feeds to flag non-compliant riders
🚔 Safety Enforcement     →  AI-assisted helmet violation detection for law enforcement
🏍️ Rider Compliance       →  Generate compliance reports for urban planning & research
```

---

## 🧠 Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **Detection Model** | YOLOv8 (Ultralytics) |
| **Web Framework** | Streamlit |
| **Computer Vision** | OpenCV |
| **Image Processing** | Pillow (PIL) |
| **Numerical Computing** | NumPy |
| **Version Control** | Git & GitHub |

---

## 📂 Project Structure

```
helmet-detection-system/
│
├── 📁 model/
│   └── best.pt                  # Custom trained YOLOv8 weights
│
├── 📁 data/
│   ├── 📁 images/               # Training & test images
│   └── data.yaml                # Dataset configuration
│
├── 📁 runs/
│   └── 📁 detect/               # YOLO training outputs & metrics
│
├── app.py                       # Streamlit dashboard entry point
├── detect.py                    # Core detection logic
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/helmet-detection-system.git
cd helmet-detection-system
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

> The app will open automatically at `http://localhost:8501`

---

## 🔄 Detection Pipeline

```
📥 Image Input
      │
      ▼
🔧 Preprocessing         →  Resize, normalize, format conversion (OpenCV + Pillow)
      │
      ▼
🤖 YOLOv8 Inference      →  Custom model detects objects, outputs bounding boxes + confidence
      │
      ▼
🎨 Annotation            →  Color-coded bounding boxes drawn on the image
      │
      ▼
📊 Results Display       →  Helmet / No Helmet counts shown in the dashboard
      │
      ▼
⬇️  Export               →  Download annotated image
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Model** | YOLOv8 Custom |
| **Classes** | `Helmet`, `No Helmet` |
| **Input Format** | JPG, PNG, WEBP |
| **Output** | Annotated image + detection count |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch → `git checkout -b feature/AmazingFeature`
3. **Commit** your changes → `git commit -m 'Add AmazingFeature'`
4. **Push** to the branch → `git push origin feature/AmazingFeature`
5. **Open** a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

Made with ❤️ using **YOLOv8** & **Streamlit**

⭐ **Star this repo if you found it helpful!**

</div>
