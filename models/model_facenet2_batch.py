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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MTCNN per Face Detection ---
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
try:
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("FaceNet model loaded successfully")
except Exception as e:
    print(f"Errore nel caricamento del modello FaceNet: {e}")
    exit()

def load_and_preprocess_images_batch(image_paths, target_size=(160, 160)):
    """Load and preprocess images in batch"""
    images = []
    valid_paths = []
    
    print(f"Loading {len(image_paths)} images...")
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transforms.functional.to_tensor(img)
            img_tensor = transforms.functional.resize(img_tensor, target_size)
            images.append(img_tensor)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    return images, valid_paths

def detect_faces_batch(images, image_paths, mtcnn_detector, batch_size=16):
    """Detect faces in batch using MTCNN"""
    face_tensors = []
    valid_paths = []
    
    print(f"Detecting faces in {len(images)} images...")
    
    # Process in smaller batches for MTCNN
    for i in tqdm(range(0, len(images), batch_size), desc="Face detection"):
        batch_images = images[i:i+batch_size]
        batch_paths = image_paths[i:i+batch_size]
        
        # Convert tensors back to PIL for MTCNN
        pil_images = []
        for img_tensor in batch_images:
            pil_img = transforms.functional.to_pil_image(img_tensor)
            pil_images.append(pil_img)
        
        try:
            # MTCNN batch processing
            detected_faces = mtcnn_detector(pil_images)
            
            for j, face in enumerate(detected_faces):
                if face is not None:
                    face_tensors.append(face)
                    valid_paths.append(batch_paths[j])
                else:
                    # Fallback: use original resized image
                    face_tensors.append(batch_images[j])
                    valid_paths.append(batch_paths[j])
                    
        except Exception as e:
            print(f"Error in batch face detection: {e}")
            # Fallback: use original images
            for j, img_tensor in enumerate(batch_images):
                face_tensors.append(img_tensor)
                valid_paths.append(batch_paths[j])
    
    return face_tensors, valid_paths

def augment_images_batch(image_tensors, num_augmentations=3):
    """Apply data augmentation to batch of images"""
    augmented_images = []
    
    for img_tensor in image_tensors:
        # Original image
        augmented_images.append(img_tensor)
        
        # Generate augmentations
        for _ in range(num_augmentations):
            aug_img = img_tensor.clone()
            
            # Random horizontal flip (50% chance)
            if np.random.random() > 0.5:
                aug_img = torch.flip(aug_img, [2])
            
            # Random rotation (-5 to 5 degrees)
            angle = np.random.uniform(-5, 5)
            aug_img = transforms.functional.rotate(aug_img, angle)
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.9, 1.1)
            aug_img = transforms.functional.adjust_brightness(aug_img, brightness_factor)
            
            augmented_images.append(aug_img)
    
    return augmented_images

def extract_embeddings_batch(model, image_tensors, batch_size=32):
    """Extract embeddings from image tensors in batch"""
    embeddings = []
    model.eval()
    
    print(f"Extracting embeddings from {len(image_tensors)} images...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_tensors), batch_size), desc="Embedding extraction"):
            batch_tensors = image_tensors[i:i+batch_size]
            
            # Stack tensors into batch
            batch = torch.stack(batch_tensors).to(device)
            
            # Extract embeddings
            batch_embs = model(batch).cpu().numpy()
            embeddings.extend(batch_embs)
    
    return np.array(embeddings).astype("float32")

def get_feature_extractor_optimized(base_model, use_mtcnn=True, use_augmentation=False, batch_size=32):
    """Optimized feature extractor with batch processing"""
    def extractor(image_paths, is_training=False):
        if not image_paths:
            return np.array([]).astype("float32"), []
        
        # Step 1: Load and preprocess images
        images, valid_paths = load_and_preprocess_images_batch(image_paths)
        
        if not images:
            return np.array([]).astype("float32"), []
        
        # Step 2: Face detection (if enabled)
        if use_mtcnn:
            face_tensors, processed_paths = detect_faces_batch(images, valid_paths, mtcnn)
        else:
            face_tensors = images
            processed_paths = valid_paths
        
        # Step 3: Data augmentation for training
        if use_augmentation and is_training:
            print("Applying data augmentation...")
            augmented_tensors = augment_images_batch(face_tensors, num_augmentations=3)
            
            # Extract embeddings from augmented images
            all_embeddings = extract_embeddings_batch(base_model, augmented_tensors, batch_size)
            
            # Average embeddings for each original image (4 variants per image)
            final_embeddings = []
            for i in range(len(face_tensors)):
                start_idx = i * 4  # 1 original + 3 augmentations
                end_idx = start_idx + 4
                avg_embedding = np.mean(all_embeddings[start_idx:end_idx], axis=0)
                final_embeddings.append(avg_embedding)
            
            embeddings = np.array(final_embeddings).astype("float32")
        else:
            # Step 4: Extract embeddings without augmentation
            embeddings = extract_embeddings_batch(base_model, face_tensors, batch_size)
        
        return embeddings, processed_paths
    
    return extractor

def build_training_embeddings_optimized(train_folder, feature_extractor_fn):
    """Build embeddings for training data organized by identity - optimized version"""
    identity_embeddings = {}
    identity_paths = {}
    
    print("Collecting training images by identity...")
    all_images_by_identity = {}
    
    # Collect all images by identity first
    for identity_folder in os.listdir(train_folder):
        identity_path = os.path.join(train_folder, identity_folder)
        if not os.path.isdir(identity_path):
            continue
            
        image_files = [
            os.path.join(identity_path, f) 
            for f in os.listdir(identity_path) 
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        if image_files:
            all_images_by_identity[identity_folder] = image_files
    
    print(f"Found {len(all_images_by_identity)} identities")
    
    # Process each identity
    for identity_name, image_files in tqdm(all_images_by_identity.items(), desc="Processing identities"):
        embs, processed_paths = feature_extractor_fn(image_files, is_training=True)
        
        if len(embs) > 0:
            # Average embeddings for this identity
            avg_embedding = np.mean(embs, axis=0)
            identity_embeddings[identity_name] = avg_embedding
            identity_paths[identity_name] = processed_paths
    
    return identity_embeddings, identity_paths

def enhanced_retrieval_with_training(query_embs, query_files, gallery_embs, gallery_files, 
                                   identity_embeddings, k=10, use_reranking=True):
    """Enhanced retrieval using training data knowledge"""
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Embedding vuoti. Retrieval non eseguibile.")
        return []

    print("Normalizing embeddings...")
    # Normalizzazione L2
    faiss.normalize_L2(gallery_embs)
    faiss.normalize_L2(query_embs)
    
    # Build identity embeddings matrix
    identity_names = []
    identity_embs = None
    if identity_embeddings:
        identity_names = list(identity_embeddings.keys())
        identity_embs = np.array([identity_embeddings[name] for name in identity_names]).astype("float32")
        faiss.normalize_L2(identity_embs)
    
    print("Building FAISS indexes...")
    # Standard gallery search
    dim = gallery_embs.shape[1]
    gallery_index = faiss.IndexFlatIP(dim)
    gallery_index.add(gallery_embs)
    
    # Identity search if available
    identity_index = None
    if identity_embeddings:
        identity_index = faiss.IndexFlatIP(dim)
        identity_index.add(identity_embs)
    
    results = []
    
    print("Performing retrieval...")
    for i in tqdm(range(len(query_files)), desc="Retrieval"):
        query_emb = query_embs[i:i+1]
        query_path = query_files[i]
        
        # Search in gallery
        gallery_distances, gallery_indices = gallery_index.search(query_emb, min(k*2, gallery_embs.shape[0]))
        
        # Search in identity embeddings for additional context
        if identity_index is not None and use_reranking:
            identity_distances, identity_indices = identity_index.search(query_emb, min(5, len(identity_names)))
            
            # Get top matching identities
            top_identities = [identity_names[idx] for idx in identity_indices[0]]
            
            # Re-rank gallery results based on identity similarity
            gallery_matches = []
            for j, idx in enumerate(gallery_indices[0]):
                gallery_path = gallery_files[idx]
                gallery_filename = os.path.basename(gallery_path)
                
                # Check if gallery image belongs to a top-matching identity
                identity_bonus = 0
                for rank, identity in enumerate(top_identities):
                    if identity.lower() in gallery_filename.lower():
                        identity_bonus = 0.1 * (1 - rank/len(top_identities))  # Bonus decreases with rank
                        break
                
                original_score = gallery_distances[0][j]
                adjusted_score = original_score + identity_bonus
                
                gallery_matches.append((gallery_path, adjusted_score))
            
            # Sort by adjusted score
            gallery_matches.sort(key=lambda x: x[1], reverse=True)
            final_matches = [match[0] for match in gallery_matches[:k]]
        else:
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

def submit(results, groupname, url="http://tatooine.disi.unitn.it:3001/retrieval/"):
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

# === CONFIGURAZIONE PATHS ===
train_folder = "train"
query_folder = "test/query"
gallery_folder = "test/gallery"

# === CARICAMENTO FILE ===
try:
    query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
except Exception as e:
    print(f"Errore nel caricamento immagini: {e}")
    exit()

if not query_files or not gallery_files:
    print("Query o gallery vuote.")
    exit()

print(f"Trovate {len(query_files)} query e {len(gallery_files)} gallery.")

# === ESTRAZIONE FEATURE OTTIMIZZATA ===
print("\n=== STARTING OPTIMIZED FEATURE EXTRACTION ===")

# Configure batch sizes based on GPU memory
# Adjust these values based on your GPU memory
FACE_DETECTION_BATCH_SIZE = 16  # For MTCNN
EMBEDDING_BATCH_SIZE = 64      # For FaceNet embeddings

feature_extractor_fn = get_feature_extractor_optimized(
    model, 
    use_mtcnn=True, 
    use_augmentation=True, 
    batch_size=EMBEDDING_BATCH_SIZE
)

# Build training embeddings
print("\n1. Building training embeddings...")
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
identity_embeddings, identity_paths = build_training_embeddings_optimized(train_folder, feature_extractor_fn)
end_time.record()
torch.cuda.synchronize()

training_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
print(f"Training completed in {training_time:.2f} seconds")
print(f"Created embeddings for {len(identity_embeddings)} identities")

# Extract query embeddings
print("\n2. Extracting query embeddings...")
start_time.record()
query_embs, processed_query_files = feature_extractor_fn(query_files, is_training=False)
end_time.record()
torch.cuda.synchronize()

query_time = start_time.elapsed_time(end_time) / 1000
print(f"Query extraction completed in {query_time:.2f} seconds")

# Extract gallery embeddings
print("\n3. Extracting gallery embeddings...")
start_time.record()
gallery_embs, processed_gallery_files = feature_extractor_fn(gallery_files, is_training=False)
end_time.record()
torch.cuda.synchronize()

gallery_time = start_time.elapsed_time(end_time) / 1000
print(f"Gallery extraction completed in {gallery_time:.2f} seconds")

print(f"\n=== TIMING SUMMARY ===")
print(f"Training: {training_time:.2f}s")
print(f"Query: {query_time:.2f}s") 
print(f"Gallery: {gallery_time:.2f}s")
print(f"Total: {training_time + query_time + gallery_time:.2f}s")

# === RETRIEVAL E SALVATAGGIO ===
if query_embs.shape[0] > 0 and gallery_embs.shape[0] > 0:
    print("\n4. Performing enhanced retrieval...")
    submission_list = enhanced_retrieval_with_training(
        query_embs, processed_query_files, 
        gallery_embs, processed_gallery_files, 
        identity_embeddings, k=10, use_reranking=True
    )
    
    data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }
    
    submission_path = "submission/submission_facenet2_optimized.py"
    save_submission_d(data, submission_path)
    
    print("\n5. Submitting results...")
    accuracy = submit(data, "Pretty Figure")
    
    print(f"\n✅ Submission salvata in: {submission_path}")
    if accuracy:
        print(f"🎯 Accuracy ottenuta: {accuracy}")
else:
    print("Embedding non estratti. Retrieval saltato.")