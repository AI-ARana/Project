import cv2
import torch
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', 'yolov5x' for different sizes

# Replace with your IP camera's RTSP URL
rtsp_url = 'rtsp://username:password@ip_address:554/stream'

# Create a Tkinter window
window = tk.Tk()
window.title("YOLOv5 Object Detection")

# Create labels to show video and report
video_label = tk.Label(window)
video_label.pack(side=tk.LEFT)

report_frame = tk.Frame(window)
report_frame.pack(side=tk.RIGHT, padx=10)

report_label = tk.Label(report_frame, text="Detection Report", font=("Helvetica", 16, "bold"))
report_label.pack()

# Create a text widget for the report
report_text = tk.Text(report_frame, height=15, width=30, font=("Helvetica", 12))
report_text.pack()

# Function to update video feed and report
def update_frame():
    global cap
    ret, frame = cap.read()
    if ret:
        # Perform detection
        results = model(frame)
        # Render results on the frame
        frame_with_boxes = results.render()[0]
        
        # Count objects
        class_counts = {cls: 0 for cls in model.names}
        for det in results.pred[0]:
            for det_class in det:
                class_id = int(det_class[5])
                class_name = model.names[class_id]
                class_counts[class_name] += 1
        
        # Update report
        report_text.delete(1.0, tk.END)  # Clear existing text
        for cls, count in class_counts.items():
            if count > 0:
                report_text.insert(tk.END, f"{cls}: {count}\n")
        
        # Convert frame to ImageTk format
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    if running:
        window.after(10, update_frame)  # Update every 10 ms

# Start capturing video from IP camera
cap = cv2.VideoCapture(rtsp_url)
running = True

# Create a function to stop the video feed
def stop_video():
    global running
    running = False
    cap.release()
    window.destroy()

# Add a Stop button
stop_button = Button(window, text="Stop", command=stop_video, fg="white", bg="red", font=("Helvetica", 12, "bold"), width=10, height=2)
stop_button.pack()

# Start updating frames
update_frame()

# Run the Tkinter event loop
window.mainloop()
