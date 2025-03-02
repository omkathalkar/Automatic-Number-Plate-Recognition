import gradio as gr
import cv2
import numpy as np
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
from collections import deque
import re

# Load YOLO model
model = YOLO("./best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Cache for storing recently processed images (Reduced to 2 for faster processing)
recent_images = deque(maxlen=3)

# Define confusion mapping for common OCR errors
digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B', '9': 'P'}
letter_to_digit = {v: k for k, v in digit_to_letter.items()}

def extract_license_plate(ocr_result, confidence_threshold=0.7):
    """
    Extracts and cleans text from OCR output, keeping only the highest-confidence result.
    """
    if not ocr_result or not ocr_result[0]:
        return "No plate detected"

    # Keep only the highest confidence text
    best_text = ""
    max_confidence = 0

    for detection in ocr_result[0]:  
        text = detection[1][0]  
        confidence = detection[1][1]  

        if confidence > confidence_threshold and confidence > max_confidence:
            best_text = text
            max_confidence = confidence

    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', best_text).upper()
    return cleaned_text if cleaned_text else "No plate detected"

def detect_and_recognize(image):
    """Runs YOLOv11 detection, applies OCR, and overlays text."""
    start_time = time.time()
    timeout = 20  # Hard limit for processing time

    image = np.array(image)
    original_image = image.copy()
    detected_text = "No plate detected"

    # Run YOLO detection
    results = model.predict(source=image, save=False)
    detections = results[0].boxes.data

    if len(detections) == 0:
        print("No License Plate Detected by YOLO.")
        return original_image, "No plate detected by YOLO Skipping OCR"

    for detection in detections[:1]:  # Process only the first detected plate
        if time.time() - start_time > timeout:
            print("Processing took too long, terminating...")
            return original_image, "Processing Timeout"

        x1, y1, x2, y2, conf, cls = detection[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop the detected license plate
        plate_image = image[y1:y2, x1:x2]

        # Run PaddleOCR on cropped plate
        ocr_result = ocr.ocr(plate_image, cls=True)
        detected_text = extract_license_plate(ocr_result)

        # Draw the detected text on the image
        if detected_text != "No plate detected":
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_image, detected_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    recent_images.append(original_image)
    return original_image, detected_text

def show_recent_images():
    return list(recent_images)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Group-15 Automatic Number Plate Recognition (ANPR) System - Optimized")
    with gr.Row():
        image_input = gr.Image(type="pil")
        image_output = gr.Image()
    text_output = gr.Textbox(label="Detected License Plate Text")
    submit_button = gr.Button("Process Image")
    recent_button = gr.Button("Show Recent Images")
    recent_output = gr.Gallery(label="Recent Processed Images")

    submit_button.click(detect_and_recognize, inputs=image_input, outputs=[image_output, text_output])
    recent_button.click(show_recent_images, inputs=[], outputs=recent_output)

demo.launch(share=True)