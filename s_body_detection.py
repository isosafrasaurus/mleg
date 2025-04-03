import cv2
import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from IPython.display import display

def compute_face_embeddings_with_display(image_path, device, face_threshold=[0.9, 0.9, 0.9]):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    mtcnn = MTCNN(keep_all=True, device=device, thresholds=face_threshold)

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    boxes, _ = mtcnn.detect(pil_image)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No faces detected in the image.")

    aligned_faces = mtcnn(pil_image)
    if aligned_faces is None:
        raise ValueError("Failed to get aligned faces.")
    if aligned_faces.ndim == 3:
        aligned_faces = aligned_faces.unsqueeze(0)

    embeddings = []
    for i in range(aligned_faces.shape[0]):
        face_tensor = aligned_faces[i].unsqueeze(0)
        with torch.no_grad():
            emb = resnet(face_tensor.to(device))
        embeddings.append(emb.cpu().numpy()[0])

    annotated_image = pil_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    for idx, box in enumerate(boxes):
        box = list(map(int, box))
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = str(idx)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    display(annotated_image)
    return embeddings

embeddings = compute_face_embeddings_with_display("/content/1pV4h1ELPxumodBcWbBAZTe5Oj4GPxorB.jpg", device)
