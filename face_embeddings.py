import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []

for img_path in image_paths:
    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except Exception as e:
        print(f"Could not open {img_path}: {e}")
        continue

    boxes, probs = mtcnn.detect(img)
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline="red", width=2)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Faces detected in {os.path.basename(img_path)}")
    plt.axis("off")
    plt.show()

    face = mtcnn(img)
    if face is None:
        print(f"No face detected in {img_path} for embedding extraction.")
        continue

    # If multiple faces are detected, mtcnn returns the first aligned face.
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face)
    embeddings.append(embedding.cpu().numpy()[0])

embeddings = np.array(embeddings)
print(f"Extracted embeddings for {embeddings.shape[0]} face(s) from {len(image_paths)} image(s).")

if embeddings.shape[0] == 0:
    print("No faces detected across images.")
else:
    clustering = DBSCAN(eps=0.8, min_samples=2, metric='euclidean').fit(embeddings)
    labels = clustering.labels_

    unique_face_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Estimated number of unique faces: {unique_face_clusters}")
