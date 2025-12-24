# import cv2
# import numpy as np
# import pickle
# from mtcnn import MTCNN
# from keras_facenet import FaceNet
# from sklearn.metrics.pairwise import cosine_similarity

# # Load models once
# detector = MTCNN()
# embedder = FaceNet()

# # Load embeddings
# with open("embeddings/face_embeddings.pkl", "rb") as f:
#     known_embeddings, known_names = pickle.load(f)

# THRESHOLD = 0.6

# def recognize_face_from_image(image_bytes):
#     # Decode image
#     np_img = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces = detector.detect_faces(rgb)
#     results = []

#     for face in faces:
#         x, y, w, h = face["box"]
#         x, y = abs(x), abs(y)

#         face_img = rgb[y:y+h, x:x+w]
#         face_img = cv2.resize(face_img, (160, 160))
#         face_img = np.expand_dims(face_img.astype("float32"), axis=0)

#         embedding = embedder.embeddings(face_img)

#         sims = cosine_similarity(embedding, known_embeddings)[0]
#         idx = np.argmax(sims)

#         if sims[idx] > THRESHOLD:
#             results.append(known_names[idx])
#         else:
#             results.append("Unknown")

#     return results


import os
import pickle
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBED_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "face_embeddings.pkl"
)


with open(EMBED_PATH, "rb") as f:
    data = pickle.load(f)

if isinstance(data, tuple):
    embeddings, names = data
    known_embeddings = dict(zip(names, embeddings))
elif isinstance(data, dict):
    known_embeddings = data
else:
    raise ValueError("Unsupported embeddings format")

def recognize_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return "Invalid image"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return "No face detected"

    x, y, w, h = faces[0]["box"]

    # Fix negative coordinates
    x, y = max(0, x), max(0, y)
    face = rgb[y:y + h, x:x + w]

    if face.size == 0:
        return "Face crop failed"

    face = cv2.resize(face, (160, 160))
    embedding = embedder.embeddings([face])[0]

    min_dist = float("inf")
    identity = "Unknown"

    THRESHOLD = 0.9

    for name, emb in known_embeddings.items():
        dist = np.linalg.norm(emb - embedding)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > THRESHOLD:
        return "Unknown"

    return identity

