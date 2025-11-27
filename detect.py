from ultralytics import YOLO
import cv2
import sys

def detect_image(image_path, output_path="output.jpg"):
    model = YOLO("yolov8s.pt")
    results = model(image_path)[0]
    img = cv2.imread(image_path)
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf)
        cls = int(box.cls)
        label = f"{results.names[cls]} {conf:.2f}"
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(img,label,(int(x1),int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imwrite(output_path, img)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image>")
        sys.exit(1)
    detect_image(sys.argv[1])
