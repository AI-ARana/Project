import cv2
import torch
import tkinter as tk
from tkinter import Button, Label, Frame
from PIL import Image, ImageTk

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)  # Load model to GPU

# Create a Tkinter window
window = tk.Tk()
window.title("YOLOv5 Object Detection")

# Create a frame for video display
video_frame = Frame(window)
video_frame.pack(side=tk.LEFT, padx=10)

# Create a label to show video
video_label = Label(video_frame)
video_label.pack()

# Create a frame for report display
report_frame = Frame(window)
report_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

# Create a label and text widget for the report
report_label = Label(report_frame, text="Detection Report", font=("Helvetica", 16, "bold"))
report_label.pack()

report_text = tk.Text(report_frame, height=20, width=40, font=("Helvetica", 12))
report_text.pack()

# Initialize class names and counts
class_names = model.names
class_counts = {name: 0 for name in class_names}

# Function to update video feed and report
def update_frame():
    global cap
    ret, frame = cap.read()
    if ret:
        # Perform detection
        results = model(frame)
        
        # Convert results to a format compatible with YOLOv5
        detections = results.xyxy[0].cpu().numpy()  # Convert to numpy array for easier handling
        
        # Debug: Print detections
        print("Detections:", detections)

        # Reset local class counts
        local_class_counts = {name: 0 for name in class_names}

        # Count objects
        for det in detections:
            if len(det) >= 6:  # Ensure the detection has class_id
                class_id = int(det[5])
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                    if class_name in local_class_counts:
                        local_class_counts[class_name] += 1

        # Debug: Print local class counts
        print("Local Class Counts:", local_class_counts)

        # Update global class_counts with local counts
        global class_counts
        class_counts = local_class_counts

        # Update report
        report_text.delete(1.0, tk.END)  # Clear existing text
        for cls, count in class_counts.items():
            if count > 0:
                report_text.insert(tk.END, f"{cls}: {count}\n")

        # Render results on the frame
        frame_with_boxes = results.render()[0]
        # Convert frame to ImageTk format
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new image
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    if running:
        window.after(10, update_frame)  # Update every 10 ms

# Start capturing video from default camera
cap = cv2.VideoCapture(0)  # 0 for default camera, or replace with video file path
running = True

# Create a function to stop the video feed
def stop_video():
    global running
    running = False
    cap.release()
    window.destroy()

# Add a Stop button
stop_button = Button(window, text="Stop", command=stop_video, fg="white", bg="red", font=("Helvetica", 12, "bold"), width=10, height=2)
stop_button.pack(pady=10)

# Start updating frames
update_frame()

# Run the Tkinter event loop
window.mainloop()
