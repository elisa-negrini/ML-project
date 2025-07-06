import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import faiss
import requests
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MTCNN for Face Detection ---
# Initialize MTCNN for face detection and alignment.
# image_size: output image size (160x160 for FaceNet).
# margin: adds margin to the detected face region.
# min_face_size: minimum size of a face to be detected.
# thresholds: detection thresholds for the three MTCNN stages.
# factor: scale factor for image pyramid.
# post_process: applies post-processing (e.g., facial landmark alignment).
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True,
    device=device
)

# --- FaceNet Model Loading ---
# Load the InceptionResnetV1 model pre-trained on VGGFace2 dataset.
# Set to evaluation mode and move to the selected device.
try:
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("FaceNet model loaded successfully")
except Exception as e:
    print(f"Error loading FaceNet model: {e}")
    exit()

def detect_and_crop_face(image_path, mtcnn_detector):
    """Detect and crop face from image using MTCNN"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Detect face and get cropped image
        img_cropped = mtcnn_detector(img)
        
        if img_cropped is not None:
            return img_cropped
        else:
            # Fallback: resize original image if no face detected
            img_tensor = transforms.functional.to_tensor(img)
            img_tensor = transforms.functional.resize(img_tensor, (160, 160))
            return img_tensor
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def augment_image(img_tensor):
    """Apply data augmentation to face image"""
    augmentations = []
    
    # Original image
    augmentations.append(img_tensor)
    
    # Horizontal flip
    augmentations.append(torch.flip(img_tensor, [2]))
    
    # Slight rotation (-5 to 5 degrees)
    angle = np.random.uniform(-5, 5)
    augmentations.append(transforms.functional.rotate(img_tensor, angle))
    
    # Brightness adjustment
    brightness_factor = np.random.uniform(0.9, 1.1)
    augmentations.append(transforms.functional.adjust_brightness(img_tensor, brightness_factor))
    
    return augmentations

def get_feature_extractor_improved(base_model, use_mtcnn=True, use_augmentation=True):
    """
    Returns a feature extractor function that uses the base model,
    with optional MTCNN face detection and data augmentation.
    """
    def extractor(image_paths, is_training=False):
        embs = []
        processed_paths = []
        
        for path in tqdm(image_paths, desc="Extracting features (FaceNet Enhanced)"):
            try:
                if use_mtcnn:
                    img_tensor = detect_and_crop_face(path, mtcnn)
                else:
                    img = Image.open(path).convert("RGB")
                    img_tensor = transforms.functional.to_tensor(img)
                    img_tensor = transforms.functional.resize(img_tensor, (160, 160))
                
                if img_tensor is None:
                    continue
                
                img_tensor = img_tensor.to(device)
                
                # Apply augmentation for training data
                if use_augmentation and is_training:
                    augmented_images = augment_image(img_tensor)
                    embeddings_for_image = []
                    
                    for aug_img in augmented_images:
                        with torch.no_grad():
                            emb = base_model(aug_img.unsqueeze(0)).cpu().numpy()[0]
                            embeddings_for_image.append(emb)
                    
                    # Average the embeddings from augmented images
                    avg_emb = np.mean(embeddings_for_image, axis=0)
                    embs.append(avg_emb)
                else:
                    with torch.no_grad():
                        emb = base_model(img_tensor.unsqueeze(0)).cpu().numpy()[0]
                        embs.append(emb)
                
                processed_paths.append(path)
                
            except Exception as e:
                print(f"Error with {path}: {e}")
                continue
                
        return np.array(embs).astype("float32") if embs else np.array([]).astype("float32"), processed_paths
    
    return extractor

def enhanced_retrieval(query_embs, query_files, gallery_embs, gallery_files, k=10):
    """Performs enhanced image retrieval using FAISS for similarity search
    """
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Empty embeddings. Retrieval cannot be performed.")
        return []

    # Normalizzazione L2
    faiss.normalize_L2(gallery_embs)
    faiss.normalize_L2(query_embs)

    # Standard gallery search
    dim = gallery_embs.shape[1]
    gallery_index = faiss.IndexFlatIP(dim)
    gallery_index.add(gallery_embs)
    
    results = []
    
    for i, query_path in enumerate(query_files):
        query_emb = query_embs[i:i+1]
        
        # Search in gallery
        gallery_distances, gallery_indices = gallery_index.search(query_emb, min(k, gallery_embs.shape[0]))

        final_matches = [gallery_files[idx] for idx in gallery_indices[0][:k]]
        
        query_rel = query_path.replace("\\", "/")
        final_matches_rel = [match.replace("\\", "/") for match in final_matches]
        
        results.append({
            "filename": query_rel,
            "gallery_images": final_matches_rel
        })
    
    return results

def save_submission_d(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            f.write(f'    "{key}": {json.dumps(value)},\n')
        f.write("}\n")

def submit(results, groupname, url=" http://tatooine.disi.unitn.it:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
        return result['accuracy']
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        return None

# === PATH CONFIGURATION ===
query_folder = "images_competition/test/query"
gallery_folder = "images_competition/test/gallery"

# === FILE LOADING ===
try:
    query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

if not query_files or not gallery_files:
    print("Query or gallery is empty. Exiting.")
    exit()

print(f"Found {len(query_files)} queries and {len(gallery_files)} gallery images.")

# === FEATURE EXTRACTION WITH ENHANCEMENTS ===
feature_extractor_fn = get_feature_extractor_improved(model, use_mtcnn=False, use_augmentation=True)

# Extract query and gallery embeddings
print("Extracting query embeddings...")
query_embs, processed_query_files = feature_extractor_fn(query_files, is_training=False)

print("Extracting gallery embeddings...")
gallery_embs, processed_gallery_files = feature_extractor_fn(gallery_files, is_training=False)

# === RETRIEVAL AND SAVING ===
if query_embs.shape[0] > 0 and gallery_embs.shape[0] > 0:
    print("Performing enhanced retrieval...")
    submission_list = enhanced_retrieval(
        query_embs, processed_query_files, 
        gallery_embs, processed_gallery_files, 
        k=10
    )
    
    data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }
    
    submission_path = "submission/submission_facenet_abl_mtcnn.py"
    save_submission_d(data, submission_path)
    
    # accuracy = submit(data, "Pretty Figure")
    
    # print(f"✅ Submission saved to: {submission_path}")
    # if accuracy is not None:
    #     print(f"🎯 Achieved accuracy: {accuracy}")
else:
    print("Embeddings not extracted. Retrieval skipped.")