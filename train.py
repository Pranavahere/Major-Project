import os
import cv2
import numpy as np
from keras_facenet import FaceNet
import pickle

# Initialize FaceNet
embedder = FaceNet()

dataset_dir = 'dataset_resized'
embeddings_dir = 'embeddings'
os.makedirs(embeddings_dir, exist_ok=True)

# Store embeddings
all_embeddings = {}
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    
    embeddings = []
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = embedder.extract(img, threshold=0.95)
        if len(faces) > 0:
            embeddings.append(faces[0]['embedding'])

    if embeddings:
        all_embeddings[person_name] = np.mean(embeddings, axis=0)

# Save all embeddings
with open(os.path.join(embeddings_dir, 'embeddings.pkl'), 'wb') as f:
    pickle.dump(all_embeddings, f)

print("✅ Embeddings generated and saved!")
