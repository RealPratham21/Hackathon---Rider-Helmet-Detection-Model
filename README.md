# 🛵 Helmet Detection using YOLOv8
Credits: [Prathamesh Bhamare](https://www.linkedin.com/in/prathamesh-bhamare-7480b52b2/) & [Moiz Shaikh](https://www.linkedin.com/in/moiz-shaikh-56471b295/)

An AI-powered object detection system designed to identify motorcyclists and detect helmet usage in real-time video streams. Built using the YOLOv8 architecture, this project showcases a practical application of deep learning for road safety monitoring and smart surveillance systems.

---

## 🚀 Demo

Try the deployed web app here: [Streamlit App](https://rider-helmet-detection-prathamesh.streamlit.app/)  
(Note: Replace with actual URL once deployed)

---

## 📌 Features

- 🎯 Detects helmets and motorcyclists in uploaded videos
- 🧠 Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 📹 Supports MP4, AVI, and MKV video formats
- 💡 Real-time bounding box visualization with confidence scores
- 📊 Frame-by-frame progress feedback
- 🖥 Streamlit-based UI for a smooth web experience

---

## 📁 Project Structure
```
📦 helmet-detection-yolov8
├── 📄 deploy1.py              # Main Streamlit application
├── 📄 new_best.pt             # YOLOv8 trained weights
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md               # Project documentation
└── 📁 .streamlit              # (Optional) Streamlit config folder
    └── 📄 config.toml         # (Optional) Custom Streamlit settings
```

---

## 🧠 Model Details

- **Architecture**: YOLOv8 (custom-trained)
- **Classes Detected**: `helmet`, `motorcycle`, `person` (mapped appropriately during training)
- **Tracking**: Integrated BYTETracker for object persistence
- **Dataset**: Derived from [Hard Hat Workers Dataset](https://public.roboflow.com/object-detection/hard-hat-workers), restructured to suit the helmet detection use case

> ⚠️ Note: Model performance is a work-in-progress. Future improvements will focus on dataset quality, class balancing, and tracker optimization.

---

## 📦 Installation & Usage (Local)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/helmet-detection-yolov8.git
   cd helmet-detection-yolov8
   ```
   
2. **Install dependencies**
   ```
   bash
   Copy
   Edit
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**
   ```
   streamlit run deploy1.py
   ```
4. **Upload a video and watch detections in real time.**

## 🤖 Tech Stack

- Python 3.11
- YOLOv8
- OpenCV
- Streamlit
- NumPy
- LAP Solver (lap module)

## 🧪 Future Improvements

- Improve model accuracy with a larger, balanced dataset
- Add separate detection for riders without helmets
- Display analytics (e.g., violation count)
- Enable real-time webcam support
- Optimize frame rate for smoother playback
