import cv2
import torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from IPython.display import display

def compute_face_embeddings_with_display(image_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert BGR (OpenCV) to RGB and create a PIL image for MTCNN
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Initialize InceptionResnetV1 for computing embeddings
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Detect face boxes and aligned faces
    boxes, _ = mtcnn.detect(pil_image)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No faces detected in the image.")

    # Get aligned face tensors; if only one face is detected, ensure batch dimension exists.
    aligned_faces = mtcnn(pil_image)
    if aligned_faces is None:
        raise ValueError("Failed to get aligned faces.")
    if aligned_faces.ndim == 3:
        aligned_faces = aligned_faces.unsqueeze(0)
    
    embeddings = []
    for i in range(aligned_faces.shape[0]):
        face_tensor = aligned_faces[i].unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            emb = resnet(face_tensor.to(device))
        embeddings.append(emb.cpu().numpy()[0])
    
    # Convert image to PIL for annotation
    annotated_image = pil_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # If available, load a default font.
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Draw each detected face with bounding box and index number.
    for idx, box in enumerate(boxes):
        box = list(map(int, box))
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = str(idx)
        text_size = draw.textsize(text, font=font)
        # Draw a filled rectangle behind text for readability
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="red")
        draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)
    
    # Display the annotated image.
    display(annotated_image)
    
    return embeddings


