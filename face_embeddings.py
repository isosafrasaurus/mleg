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
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
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
