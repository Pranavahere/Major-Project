import cv2
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
from datetime import datetime
import csv
import winsound
import time


# Load FaceNet
embedder = FaceNet()

# Load embeddings
with open('embeddings/embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)

attendance_dir = 'attendance'
unknown_dir = 'unknown_faces'
os.makedirs(attendance_dir, exist_ok=True)
os.makedirs(unknown_dir, exist_ok=True)

today_str = datetime.now().strftime("%Y-%m-%d")
attendance_file = os.path.join(attendance_dir, f"{today_str}.csv")

if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Time'])

last_attendance_time = {}

# Compare two embeddings
def find_match(face_embedding, known_embeddings, threshold=0.6):
    min_dist = float('inf')
    identity = "Unknown"
    
    for name, db_emb in known_embeddings.items():
        dist = np.linalg.norm(face_embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > threshold:
        return "Unknown"
    else:
        return identity

# Open webcam
cap = cv2.VideoCapture(0)
print("🎥 Starting real-time FaceNet recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = embedder.extract(rgb_frame, threshold=0.95)

    for face in faces:
        box = face['box']
        embedding = face['embedding']

        name = find_match(embedding, known_embeddings)

        x, y, w, h = box
        x, y = max(0, x), max(0, y)

        time_now = datetime.now()
        time_str = time_now.strftime("%H:%M:%S")
        current_time = time.time()

        if name == "Unknown":
            img_filename = f"unknown_{time_now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            img_path = os.path.join(unknown_dir, img_filename)
            cv2.imwrite(img_path, frame[y:y+h, x:x+w])
            winsound.Beep(1000, 500)
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_filename, time_str])
        else:
            last_time = last_attendance_time.get(name, 0)
            if current_time - last_time > 120:
                with open(attendance_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_str])
                last_attendance_time[name] = current_time

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    cv2.imshow("Real-time FaceNet Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
