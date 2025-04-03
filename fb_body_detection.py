import os, cv2, time, torch, numpy as np
from PIL import Image
from IPython.display import display, clear_output, Image as IPyImage
from facenet_pytorch import MTCNN, InceptionResnetV1

def process_volunteer_train(input_directory, output_directory, embedding_thresh = 0.8, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    def compute_embedding(face_tensor):
        with torch.no_grad():
            emb = resnet(face_tensor.to(device))
        return emb.cpu().numpy()[0]

    def fill_face_with_black(image, box):
        x1, y1, x2, y2 = box
        image[y1:y2, x1:x2] = 0

    def box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    os.makedirs(output_directory, exist_ok=True)

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.to(device)
    yolo_model.eval()

    mtcnn = MTCNN(keep_all=True, device=device)

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    target_face_embedding = None
    target_face_box = None

    image_files = sorted([f for f in os.listdir(input_directory) if f.lower().endswith(('.jpg', '.jpeg', '.JPEG', '.png', '.bmp', '.tiff'))])
    image_files[0] = "/content/drive/MyDrive/Pictures/mleg/sk/_witness.jpg"

    for filename in image_files:
        image_path = os.path.join(input_directory, filename)
        print("Processing", image_path)

        image = cv2.imread(image_path)
        if image is None:
            print("  Failed to load", filename)
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)

        face_boxes, _ = mtcnn.detect(pil_img)
        if face_boxes is None or len(face_boxes) == 0:
            print("  No faces detected in", filename)
            continue

        aligned_faces = mtcnn(pil_img)
        if aligned_faces is None:
            print("  No aligned faces in", filename)
            continue
        if aligned_faces.ndim == 3:  
            aligned_faces = aligned_faces.unsqueeze(0)

        face_embeddings = []
        for i in range(aligned_faces.shape[0]):
            face_tensor = aligned_faces[i].unsqueeze(0)
            emb = compute_embedding(face_tensor)
            face_embeddings.append(emb)

        if target_face_embedding is None:
            target_face_embedding = face_embeddings[0]
            target_face_box = list(map(int, face_boxes[0]))
            print("  Setting target face from", filename)

        best_distance = float('inf')
        target_index = None
        for i, emb in enumerate(face_embeddings):
            d = np.linalg.norm(emb - target_face_embedding)
            if d < best_distance:
                best_distance = d
                target_index = i

        if target_index is None or best_distance > embedding_thresh:
            print("  Target face not found in", filename)
            continue

        target_box_current = list(map(int, face_boxes[target_index]))

        for i, box in enumerate(face_boxes):
            if i != target_index:
                box_int = list(map(int, box))
                fill_face_with_black(image, box_int)

        try:
            yolo_results = yolo_model(rgb_image)
            detections = yolo_results.pandas().xyxy[0]
        except Exception as e:
            print("  YOLOv5 error:", e)
            continue

        person_boxes = []
        for _, row in detections.iterrows():
            if row['name'] == 'person':
                box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                person_boxes.append(box)

        if not person_boxes:
            print("  No person detected in", filename)
            continue

        target_center = box_center(target_box_current)
        best_person_box = None
        best_offset = float('inf')
        for p_box in person_boxes:
            x1, y1, x2, y2 = p_box
            if x1 <= target_center[0] <= x2 and y1 <= target_center[1] <= y2:
                person_center = box_center(p_box)
                offset = abs(target_center[0] - person_center[0])
                if offset < best_offset:
                    best_offset = offset
                    best_person_box = p_box

        if best_person_box is None:
            print("  No person box surrounds target face in", filename)
            continue

        mask = np.zeros(image.shape[:2], dtype=bool)
        tx1, ty1, tx2, ty2 = target_box_current
        mask[ty1:ty2, tx1:tx2] = True

        for p_box in person_boxes:

            if p_box == best_person_box:
                continue
            x1, y1, x2, y2 = p_box

            region = image[y1:y2, x1:x2]
            region_mask = mask[y1:y2, x1:x2]
            region[~region_mask] = 0
            image[y1:y2, x1:x2] = region

        FACE_MARGIN = 20

        bx1, by1, bx2, by2 = best_person_box
        tx1, ty1, tx2, ty2 = target_box_current

        if tx1 >= bx1 and ty1 >= by1 and tx2 <= bx2 and ty2 <= by2:
            crop_box = best_person_box
            print("  Cropping to person box.")
        else:
            h, w = image.shape[:2]
            crop_box = (max(0, tx1 - FACE_MARGIN),
                        max(0, ty1 - FACE_MARGIN),
                        min(w, tx2 + FACE_MARGIN),
                        min(h, ty2 + FACE_MARGIN))
            print("  Cropping to target face box with margin.")

        cx1, cy1, cx2, cy2 = crop_box
        cropped_image = image[cy1:cy2, cx1:cx2].copy()

        cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)

        clear_output(wait=True)
        temp_path = os.path.join(output_directory, "temp_display.jpg")
        cv2.imwrite(temp_path, image)
        display(IPyImage(filename=temp_path, width=400))

        output_path = os.path.join(output_directory, f"processed_{filename}")
        cv2.imwrite(output_path, cropped_image)
        print("  Saved processed image to", output_path)

        time.sleep(0.5)

process_volunteer_train("/content/drive/MyDrive/Pictures/mleg/sk", "output")
