Retail Shelf Object Detector using YOLOv8

This project uses a custom-trained YOLOv8 model to detect and count items on retail shelves from images. It provides a lightweight and interactive web app interface built with Gradio to upload images, visualize detections, and display category-wise object counts.

🚀 Features:
- YOLOv8-based detection of grocery items.
- Annotated output images with bounding boxes.
- Gradio interface for real-time inference using upload or webcam.
- Displays total count and label summary per image.
  
🧠 How It Works
- You upload or capture an image via the Gradio interface.
- The YOLOv8 model (best.pt) predicts bounding boxes and class labels.
- The app counts and displays each type of product detected.
- A result image with boxes and a label breakdown is returned.
  
🛠️ Requirements
- Python 3.8+
- YOLOv8 (Ultralytics)
- Gradio
- PIL (Pillow)
- OpenCV
  
📂 run
- run train_yolo.ipynb ( to get best.pt file )
- python retail_detector.py ( on terminal to start gradio interface )
