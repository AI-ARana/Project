import cv2
import torch
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', 'yolov5x' for different sizes

# Create a Tkinter window
window = tk.Tk()
window.title("YOLOv5 Object Detection")

# Create a label to show video
video_label = tk.Label(window)
video_label.pack()

# Function to update video feed
def update_frame():
    global cap
    ret, frame = cap.read()
    if ret:
        # Perform detection
        results = model(frame)
        # Render results on the frame
        frame_with_boxes = results.render()[0]
        # Convert frame to ImageTk format
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    if running:
        window.after(10, update_frame)  # Update every 10 ms

# Start capturing video
cap = cv2.VideoCapture(0)  # 0 for default camera, or replace with video file path
running = True

# Create a function to stop the video feed
def stop_video():
    global running
    running = False
    cap.release()
    window.destroy()

# Add a Stop button
stop_button = Button(window, text="Stop", command=stop_video)
stop_button.pack()

# Start updating frames
update_frame()

# Run the Tkinter event loop
window.mainloop()
