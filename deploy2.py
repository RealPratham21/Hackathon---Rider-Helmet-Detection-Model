import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# ---- App Configuration ----
st.set_page_config(
    page_title="Helmet Detection - Model 2",
    page_icon="🚀",
    layout="wide"
)

# ---- Custom Styling ----
st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .stMarkdown {
            text-align: center;
        }
        .team-box {
            text-align: center;
            font-size: 18px;
            border: 2px solid #4CAF50;
            padding: 10px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.1);
        }
        .team-box a {
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Branding & Team Info ----
st.markdown(
    """
    <div class='team-box'>
        <h2>🚀 Helmet Detection - YOLOv8 (Model 2)</h2>
        <p><strong>Team: Neural Nexus</strong></p>
        <p>Developed by:</p>
        <p><a href='https://www.linkedin.com/in/prathamesh-bhamare-7480b52b2/' target='_blank'>Prathamesh Bhamare</a> & 
           <a href='https://www.linkedin.com/in/moiz-shaikh-56471b295/' target='_blank'>Moiz Shaikh</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Load YOLO Model ----
st.sidebar.header("Model Configuration")
model_path = "new_best_5.pt"
st.sidebar.write(f"🛠 Model: `{model_path}`")

model = YOLO(model_path)

# ---- Upload Video File ----
st.sidebar.header("Upload Video")
uploaded_file = st.sidebar.file_uploader("Upload a video (MP4, AVI, MKV)", type=["mp4", "avi", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Load video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("🚨 Error: Could not open video file.")
    else:
        st.success("✅ Video uploaded successfully!")
        
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        st.write("🔍 **Processing Video...**")
        time.sleep(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

            # Resize frame
            frame_resized = cv2.resize(frame, (640, 480))

            # Run YOLO detection
            results = model.track(frame_resized, persist=True)

            # Draw bounding boxes
            for result in results:
                for box, label_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[int(label_id)]

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame_resized, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert to RGB for Streamlit display
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_resized, channels="RGB", use_container_width=True)

        cap.release()
        st.success("🎉 Video Processing Complete!")
