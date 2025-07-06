import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import faiss
import requests
from facenet_pytorch import InceptionResnetV1, MTCNN
import open_clip
from collections import defaultdict
import random
from datetime import datetime

# ========== DEVICE AND CONFIGURATION ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ========== PATHS AND COMPETITION CONFIGURATION ==========
# GROUP_NAME rimosso

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "..")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
GALLERY_DIR = os.path.join(BASE_DIR, "test", "gallery")
QUERY_DIR = os.path.join(BASE_DIR, "test", "query")

# Model weights path
CLIP_MODEL_PATH = os.path.join(script_dir, "clip_arcface_trained.pt")

# ========== ENSEMBLE CONFIGURATION ==========
TOTAL_SCORE = 970 + 910
FACENET_WEIGHT = 970/ TOTAL_SCORE
CLIP_WEIGHT = 910/ TOTAL_SCORE

# Confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence to consider a prediction

print(f"🎯 Ensemble Configuration:")
print(f"   - FaceNet Weight: {FACENET_WEIGHT:.3f}")
print(f"   - CLIP+ArcFace Weight: {CLIP_WEIGHT:.3f}")
print(f"   - Confidence Threshold: {MIN_CONFIDENCE_THRESHOLD}")

# ========== CLIP+ARCFACE MODEL ==========
class CLIPArcFace(nn.Module):
    def __init__(self, clip_model, embed_dim=1024, num_classes=None, unfreeze_layers=4):
        super().__init__()
        self.clip = clip_model
        
        total_layers = len(clip_model.visual.transformer.resblocks)
        print(f"[CLIP] Total transformer layers: {total_layers}, unfreezing last {unfreeze_layers}")
        
        for name, param in clip_model.visual.named_parameters():
            if 'resblocks' in name:
                try:
                    layer_idx = int(name.split('.')[2])
                    param.requires_grad = layer_idx >= total_layers - unfreeze_layers
                except:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
        clip_dim = 768
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        if num_classes:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
    
    def forward(self, x):
        clip_features = self.clip.visual(x)
        embeddings = F.normalize(self.projection(clip_features), dim=-1)
        if hasattr(self, 'classifier'):
            logits = self.classifier(embeddings)
            return embeddings, logits
        return embeddings
    
    def extract_features(self, x):
        with torch.no_grad():
            clip_feat = self.clip.visual(x)
            emb = F.normalize(self.projection(clip_feat), dim=-1)
        return emb

# ========== FACENET FUNCTIONS ==========
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

def detect_and_crop_face(image_path, mtcnn_detector):
    try:
        img = Image.open(image_path).convert('RGB')
        img_cropped = mtcnn_detector(img)
        if img_cropped is not None:
            return img_cropped
        else:
            img_tensor = transforms.functional.to_tensor(img)
            img_tensor = transforms.functional.resize(img_tensor, (160, 160))
            return img_tensor
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def augment_image(img_tensor):
    augmentations = [img_tensor]
    augmentations.append(torch.flip(img_tensor, [2]))
    angle = np.random.uniform(-5, 5)
    augmentations.append(transforms.functional.rotate(img_tensor, angle))
    brightness_factor = np.random.uniform(0.9, 1.1)
    augmentations.append(transforms.functional.adjust_brightness(img_tensor, brightness_factor))
    return augmentations

# ========== SIMPLE ENSEMBLE CLASS ==========
class SimpleCLIPFaceNetEnsemble:
    def __init__(self, facenet_model, clip_model, clip_preprocess, 
                 facenet_weight, clip_weight, confidence_threshold):
        self.facenet_model = facenet_model.eval()
        self.clip_model = clip_model.eval()
        self.clip_preprocess = clip_preprocess
        self.facenet_weight = facenet_weight
        self.clip_weight = clip_weight
        self.confidence_threshold = confidence_threshold
        print(f"✅ Simple Ensemble initialized.")
    
    def extract_facenet_embeddings(self, image_paths, is_training=False, use_augmentation=True):
        embs, processed_paths, failed_paths = [], [], []
        desc = "Extracting FaceNet embeddings" + (" (train)" if is_training else "")
        for path in tqdm(image_paths, desc=desc):
            try:
                img_tensor = detect_and_crop_face(path, mtcnn)
                if img_tensor is None:
                    failed_paths.append(path)
                    continue
                img_tensor = img_tensor.to(device)
                
                if use_augmentation and is_training:
                    augmented_images = augment_image(img_tensor)
                    embeddings_for_image = [self.facenet_model(aug_img.unsqueeze(0)).cpu().numpy()[0] for aug_img in augmented_images]
                    avg_emb = np.mean(embeddings_for_image, axis=0)
                    embs.append(avg_emb)
                else:
                    with torch.no_grad():
                        emb = self.facenet_model(img_tensor.unsqueeze(0)).cpu().numpy()[0]
                        embs.append(emb)
                processed_paths.append(path)
            except Exception as e:
                print(f"⚠️ FaceNet error with {path}: {e}")
                failed_paths.append(path)
        return np.array(embs, dtype="float32"), processed_paths, failed_paths
    
    def extract_clip_embeddings(self, image_paths):
        embs, processed_paths, failed_paths = [], [], []
        for path in tqdm(image_paths, desc="Extracting CLIP+ArcFace embeddings"):
            try:
                img_pil = Image.open(path).convert('RGB')
                img_preprocessed = self.clip_preprocess(img_pil).unsqueeze(0).to(device)
                emb = self.clip_model.extract_features(img_preprocessed)
                embs.append(emb.cpu().numpy()[0])
                processed_paths.append(path)
            except Exception as e:
                print(f"⚠️ CLIP error with {path}: {e}")
                failed_paths.append(path)
        return np.array(embs, dtype="float32"), processed_paths, failed_paths

    def simple_ensemble_retrieval(self, query_embs_fn, query_embs_clip, query_files, 
                                  gallery_embs_fn, gallery_embs_clip, gallery_files, k=10):
        print("🚀 Starting simple ensemble retrieval...")
        
        # Normalize embeddings
        if gallery_embs_fn.shape[0] > 0: faiss.normalize_L2(gallery_embs_fn)
        if query_embs_fn.shape[0] > 0: faiss.normalize_L2(query_embs_fn)
        if gallery_embs_clip.shape[0] > 0: faiss.normalize_L2(gallery_embs_clip)
        if query_embs_clip.shape[0] > 0: faiss.normalize_L2(query_embs_clip)
        
        # Build FAISS indices
        dim_fn = gallery_embs_fn.shape[1]
        gallery_index_fn = faiss.IndexFlatIP(dim_fn)
        gallery_index_fn.add(gallery_embs_fn)
        
        dim_clip = gallery_embs_clip.shape[1]
        gallery_index_clip = faiss.IndexFlatIP(dim_clip)
        gallery_index_clip.add(gallery_embs_clip)
        
        results = []
        for i in tqdm(range(len(query_files)), desc="Performing retrieval"):
            query_path = query_files[i]
            ensemble_scores = defaultdict(float)
            
            # Search directly for k results
            search_k = min(k, len(gallery_files))
            
            # FaceNet search
            query_emb_fn = query_embs_fn[i:i+1]
            fn_distances, fn_indices = gallery_index_fn.search(query_emb_fn, search_k)
            for idx, score in zip(fn_indices[0], fn_distances[0]):
                if score >= self.confidence_threshold:
                    ensemble_scores[gallery_files[idx]] += self.facenet_weight * score

            # CLIP search
            query_emb_clip = query_embs_clip[i:i+1]
            clip_distances, clip_indices = gallery_index_clip.search(query_emb_clip, search_k)
            for idx, score in zip(clip_indices[0], clip_distances[0]):
                if score >= self.confidence_threshold:
                    ensemble_scores[gallery_files[idx]] += self.clip_weight * score
            
            # Get final results
            if ensemble_scores:
                sorted_matches = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
                final_matches = [match[0] for match in sorted_matches[:k]]
            else:
                # Fallback: take first k from gallery instead of random
                final_matches = gallery_files[:k]
            
            results.append({"filename": query_path, "gallery_images": final_matches})
        
        return results

def save_submission_d(submission_dict, output_path): # GROUP_NAME rimosso dalla firma
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        # f.write(f'groupname = "{group_name}"\n') # riga rimossa
        f.write("data = {\n") # Modificato retrieval in data
        for key, value in submission_dict.items():
            
            formatted_values = [f'"{os.path.basename(v)}"' for v in value]
            f.write(f'    "{key}": [{", ".join(formatted_values)}],\n')
        f.write("}\n")

# ========== MAIN EXECUTION ==========
def main():
    print("🚀 Starting Simple CLIP+ArcFace & FaceNet Ensemble")
    print("=" * 60)
    
    # Load FaceNet model
    print("📥 Loading FaceNet model...")
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("✅ FaceNet model loaded successfully.")
    
    # Load CLIP model
    print("📥 Loading CLIP+ArcFace model...")
    clip_base, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    
    clip_model = CLIPArcFace(
        clip_base, embed_dim=1024, num_classes=None, unfreeze_layers=4
    ).to(device)
    clip_model.load_state_dict(torch.load(CLIP_MODEL_PATH, map_location=device), strict=False)    
    clip_model.eval()
    print("✅ CLIP+ArcFace model loaded successfully.")
    
    ensemble = SimpleCLIPFaceNetEnsemble(
        facenet_model, clip_model, clip_preprocess,
        FACENET_WEIGHT, CLIP_WEIGHT, MIN_CONFIDENCE_THRESHOLD
    )
    
    print("📂 Loading image files...")
    query_files = sorted([os.path.join(QUERY_DIR, f) for f in os.listdir(QUERY_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    gallery_files = sorted([os.path.join(GALLERY_DIR, f) for f in os.listdir(GALLERY_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"📊 Found {len(query_files)} query images and {len(gallery_files)} gallery images.")
    
    print("🔍 Extracting gallery embeddings...")
    gallery_embs_fn, _, _ = ensemble.extract_facenet_embeddings(gallery_files, is_training=False)
    gallery_embs_clip, _, _ = ensemble.extract_clip_embeddings(gallery_files)
    
    print("🔍 Extracting query embeddings...")
    query_embs_fn, _, _ = ensemble.extract_facenet_embeddings(query_files, is_training=False)
    query_embs_clip, _, _ = ensemble.extract_clip_embeddings(query_files)
    
    print("🎯 Performing ensemble retrieval...")
    submission_list = ensemble.simple_ensemble_retrieval(
        query_embs_fn, query_embs_clip, query_files,
        gallery_embs_fn, gallery_embs_clip, gallery_files,
        k=10
    )
    
    if submission_list:
        submission_dict = {
            os.path.basename(entry['filename']): entry['gallery_images']
            for entry in submission_list
        }
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # timestamp non più necessario per il nome file fisso
        output_filename = os.path.join(BASE_DIR, "submission_ensemble_final.py") # Nome file fisso
        
        print(f"💾 Saving submission to: {output_filename}")
        try:
            save_submission_d(submission_dict, output_filename) # Rimosso GROUP_NAME
            print("   - File saved successfully.")
            print(f"   - Ora puoi caricare manualmente il file '{output_filename}' sulla piattaforma della competizione.")
        except Exception as e:
             print(f"❌ Error saving submission file: {e}")
    else:
        print("❌ Submission list is empty. No output file generated.")
    print("=" * 60)
    print("🏁 Ensemble execution finished.")

if __name__ == '__main__':
    main()
