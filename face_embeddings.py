from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_boxes(image, boxes, title):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    if boxes is not None:
        for box in boxes:
            corner = (box[0], box[1])
            width = box[2] - box[0]
            height = box[3] - box[1]
            rect = patches.Rectangle(corner, width, height, linewidth = 1, edgecolor = 'r', facecolor = 'none')
            ax.add_patch(rect)
    plt.title(title)
    plt.show()

def get_face_embedding(image_path, threshold_pnet=0.6, threshold_rnet=0.7, threshold_onet=0.9):
    image = Image.open(image_path).convert("RGB")
    mtcnn = MTCNN(keep_all=True, thresholds=[threshold_pnet, threshold_rnet, threshold_onet])
    boxes, _ = mtcnn.detect(image)
    num_faces = 0 if boxes is None else len(boxes)
    if num_faces != 1:
        display_boxes(image, boxes, f"Detected {num_faces} faces")
        raise ValueError(f"Expected exactly one face, but found {num_faces}.")
    display_boxes(image, boxes, "Detected Face")

    # Get the aligned face
    face = mtcnn(image)
    if isinstance(face, list):
        if len(face) != 1:
            display_boxes(image, boxes, f"Multiple faces during alignment: {len(face)}")
            raise ValueError(f"Expected exactly one face after alignment, but found {len(face)}.")
        face = face[0]

    # Check if the face tensor already has a batch dimension.
    if face.ndim == 3:
        face_tensor = face.unsqueeze(0)  # shape becomes (1, 3, H, W)
    else:
        face_tensor = face  # assume it already is (1, 3, H, W)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    with torch.no_grad():
        embedding = resnet(face_tensor)

    return embedding

def get_average_face_embedding(folder_path, threshold_pnet=0.6, threshold_rnet=0.7, threshold_onet=0.9):
    embeddings = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        try:
            embedding = get_face_embedding(file_path, threshold_pnet=threshold_pnet, threshold_rnet=threshold_rnet, threshold_onet=threshold_onet)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    if not embeddings:
        raise ValueError("No valid face embeddings found in the folder.")

    # Each embedding has shape (1, 512). Concatenate along the batch dimension.
    embeddings_tensor = torch.cat(embeddings, dim=0)  # Shape (N, 512)
    average_embedding = embeddings_tensor.mean(dim=0)  # Averaged embedding (512,)

    return average_embedding
  
def match_faces_in_folder(target_embedding, folder_path, distance_threshold=0.9, threshold_pnet=0.6, threshold_rnet=0.7, threshold_onet=0.9):
    mtcnn = MTCNN(keep_all=True, thresholds=[threshold_pnet, threshold_rnet, threshold_onet])
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # Ensure target_embedding is a 1D tensor of shape (512,)
    if target_embedding.ndim == 2:
        target_embedding = target_embedding.squeeze(0)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            continue

        boxes, _ = mtcnn.detect(image)
        if boxes is None:
            print(f"No faces detected in {file_path}")
            continue

        faces = mtcnn(image)
        if faces is None:
            print(f"Could not align faces in {file_path}")
            continue
        if isinstance(faces, torch.Tensor) and faces.ndim == 3:
            faces = faces.unsqueeze(0)  # Ensure batch dimension

        with torch.no_grad():
            embeddings = resnet(faces)  # Shape: (N, 512)

        # Compute Euclidean distances between each face embedding and the target_embedding.
        distances = torch.norm(embeddings - target_embedding, dim=1)
        matches = distances < distance_threshold

        if matches.any():
            matched_distances = [f"{d.item():.2f}" for d, m in zip(distances, matches) if m]
            title = f"{filename}: Matches found with distances: " + ", ".join(matched_distances)
            match_boxes = [box for box, m in zip(boxes, matches) if m]
            display_boxes(image, match_boxes, title)
        else:
            print(f"No matching faces in {file_path}")
