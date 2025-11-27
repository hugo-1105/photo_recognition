import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter

model = YOLO("yolov8s.pt")

st.title("🍎 Image Object Recognition (YOLOv8)")
st.write("Upload an image to detect objects. Results include bounding boxes and a summary list.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = model(img)[0]

    detected_classes = []

    # Draw boxes + collect class names
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf)
        cls = int(box.cls)
        label = f"{results.names[cls]} {conf:.2f}"

        detected_classes.append(results.names[cls])

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, label, (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Display Image
    st.image(img, channels="BGR", caption="Detected Objects")

    # Display Summary
    st.subheader("📋 Detection Summary")

    if detected_classes:
        counts = Counter(detected_classes)

        st.write(f"**Total objects detected:** {len(detected_classes)}")

        st.write("### 🧾 Object Count:")
        for cls, count in counts.items():
            st.write(f"- **{cls}** × {count}")

    else:
        st.write("No objects detected.")
