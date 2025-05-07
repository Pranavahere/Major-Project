from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import numpy as np
import pickle
import time
from keras_facenet import FaceNet
from datetime import datetime
import csv
import subprocess
from collections import defaultdict, Counter

# Initialize Flask and FaceNet
app = Flask(__name__)
embedder = FaceNet()
camera = cv2.VideoCapture(0)

# Paths
dataset_dir = 'dataset_resized'
embeddings_dir = 'embeddings'
attendance_dir = 'attendance'
unknown_dir = 'unknown_faces'

# Ensure directories exist
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)
os.makedirs(attendance_dir, exist_ok=True)
os.makedirs(unknown_dir, exist_ok=True)

# Load embeddings
def load_known_embeddings():
    path = os.path.join(embeddings_dir, 'embeddings.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

known_embeddings = load_known_embeddings()
last_attendance_time = {}

# Matching function
def find_match(face_embedding, known_embeddings, threshold=0.6):
    min_dist = float('inf')
    identity = "Unknown"
    for name, db_emb in known_embeddings.items():
        dist = np.linalg.norm(face_embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name
    return identity if min_dist <= threshold else "Unknown"

# Video streaming
def generate_frames():
    global known_embeddings

    today_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file = os.path.join(attendance_dir, f"{today_str}.csv")
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time'])

    while True:
        success, frame = camera.read()
        if not success:
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Home
@app.route('/')
def home():
    return render_template('index.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        name = request.form['name']
        files = request.files.getlist('images')
        user_dir = os.path.join(dataset_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        for img in files:
            if img and '.' in img.filename:
                img.save(os.path.join(user_dir, img.filename))

        subprocess.call(['python', 'generate_embeddings.py'])
        global known_embeddings
        known_embeddings = load_known_embeddings()

        return redirect(url_for('home'))
    return render_template('register.html')

# Check-in
@app.route('/checkin')
def checkin():
    return render_template('checkin.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Admin logs
@app.route('/admin')
def admin():
    logs = sorted(os.listdir(attendance_dir), reverse=True)
    return render_template('admin.html', logs=logs)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(attendance_dir, filename)

# Analytics route (for admin.html charts)
@app.route('/admin/analytics')
def analytics():
    data = defaultdict(list)
    visitor_counts = Counter()
    files = os.listdir(attendance_dir)
    for file in files:
        if not file.endswith('.csv'): continue
        with open(os.path.join(attendance_dir, file), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                name, time_val = row
                data[file].append((name, time_val))
                visitor_counts[name] += 1

    daily_counts = {date: len(entries) for date, entries in data.items()}

    hourly_data = defaultdict(int)
    for entries in data.values():
        for _, time_val in entries:
            hour = time_val.split(':')[0]
            hourly_data[hour] += 1

    frequent_visitors = {k: v for k, v in visitor_counts.items() if v > 2}
    new_visitors = {k: v for k, v in visitor_counts.items() if v <= 2}

    return jsonify({
        'daily_counts': daily_counts,
        'hourly_data': hourly_data,
        'frequent_visitors': frequent_visitors,
        'new_visitors': new_visitors
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
