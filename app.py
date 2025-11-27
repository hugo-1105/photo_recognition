import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model = YOLO("yolov8s.pt")

st.title("🍎 Image Object Recognition (YOLOv8)")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = model(img)[0]

    for box in results.boxes:
        x1,y1,x2,y2 = box.xyxy[0]
        conf = float(box.conf)
        cls = int(box.cls)
        label = f"{results.names[cls]} {conf:.2f}"
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(img,label,(int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    st.image(img, channels="BGR", caption="Detected Objects")
