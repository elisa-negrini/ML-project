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
import open_clip
import random

BATCH_SIZE = 32
TOP_K = 10
SEED = 42

# ==== PATH ====
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "..")

GALLERY_DIR = os.path.join(BASE_DIR, "images_competition/test", "gallery")
QUERY_DIR = os.path.join(BASE_DIR, "images_competition/test", "query")

SUBMISSION_PATH = os.path.join(script_dir, "submission_clip_zero_shot.py")

# ==== DEVICE & SEED ====
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

class CLIPZeroShot(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        
        for param in clip_model.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        with torch.no_grad():
            features = self.clip.encode_image(x)
            normalized_features = F.normalize(features, dim=-1)
            return normalized_features

class FlatTestDataset(Dataset):
    def __init__(self, root_dir, preprocess_fn):
        self.root_dir = root_dir
        self.preprocess_fn = preprocess_fn
        self.samples = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"[ERROR] Directory not found: {root_dir}")
        
        for img_file in os.listdir(root_dir):
            if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                self.samples.append(os.path.join(root_dir, img_file))
        
        print(f"[INFO] Found {len(self.samples)} images in {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.preprocess_fn: 
                image = self.preprocess_fn(image)
            return image, img_path
        except Exception as e:
            print(f"[ERROR] Loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), img_path

def extract_embeddings(model, data_loader, dataset_name=""):
    print(f"[INFO] Extracting embeddings from {dataset_name}...")
    
    model.eval()
    embeddings = []
    file_paths = []
    
    with torch.no_grad():
        for images, paths_batch in tqdm(data_loader, desc=f'Processing {dataset_name}'):
            images = images.to(device)
            
            emb = model.extract_features(images)
            
            embeddings.append(emb.cpu())
            file_paths.extend(paths_batch)
    
    if not embeddings:
        print(f"[WARNING] No embeddings extracted from {dataset_name}")
        return torch.tensor([]), []
    
    all_embeddings = torch.vstack(embeddings)
    print(f"[INFO] Extracted {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
    
    return all_embeddings, file_paths

# ==== RETRIEVAL ====
def cosine_retrieval(query_embeddings, gallery_embeddings, query_files, gallery_files, k=10):
    print(f"[INFO] Performing cosine similarity retrieval (top-{k})...")
   
    query_np = query_embeddings.numpy()
    gallery_np = gallery_embeddings.numpy()
    
    similarities = query_np @ gallery_np.T 
    
    ranks = np.argsort(similarities, axis=1)[:, ::-1]
    
    results = {}
    for i, query_file in enumerate(query_files):
        top_k_indices = ranks[i][:k]
        matches = [gallery_files[idx] for idx in top_k_indices]
        results[query_file] = matches
    
    print(f"[INFO] Retrieval completed for {len(query_files)} queries")
    return results

def save_submission_dict(results, output_path):
    print(f"[INFO] Saving submission to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            formatted_key = os.path.basename(key)
            formatted_values = [os.path.basename(v) for v in value]
            f.write(f'    "{formatted_key}": {json.dumps(formatted_values)},\n')
        f.write("}\n")
    
    print(f"[INFO] Submission saved successfully!")

# ==== MAIN PIPELINE ====
def main():
    print("[INFO] === ABLATION STUDY: CLIP ZERO-SHOT BASELINE (SCRIPT 0) ===")
    print("[INFO] This script uses CLIP out-of-the-box without any training or additional layers")
    
    print("[INFO] Loading ViT-L-14 model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model.to(device)
    
    model = CLIPZeroShot(clip_model).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] All parameters are frozen (zero-shot inference only)")
    
    print("[INFO] Creating datasets...")
    try:
        query_dataset = FlatTestDataset(QUERY_DIR, preprocess)
        gallery_dataset = FlatTestDataset(GALLERY_DIR, preprocess)

        if len(query_dataset) == 0 or len(gallery_dataset) == 0:
            print("[CRITICAL ERROR] Query or Gallery dataset is empty. Exiting.")
            return

        query_loader = DataLoader(
            query_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=2
        )
        gallery_loader = DataLoader(
            gallery_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=2
        )
        
    except FileNotFoundError as e:
        print(e)
        return

    print("[INFO] === FEATURE EXTRACTION PHASE ===")
    query_embeddings, query_files = extract_embeddings(model, query_loader, "Query")
    gallery_embeddings, gallery_files = extract_embeddings(model, gallery_loader, "Gallery")
    
    if len(query_embeddings) == 0 or len(gallery_embeddings) == 0:
        print("[CRITICAL ERROR] Failed to extract embeddings. Exiting.")
        return
    
    print("[INFO] === RETRIEVAL PHASE ===")
    results = cosine_retrieval(
        query_embeddings, gallery_embeddings, 
        query_files, gallery_files, 
        k=TOP_K
    )
    
    save_submission_dict(results, SUBMISSION_PATH)
    
    print(f"\n[INFO] === EXPERIMENT SUMMARY ===")
    print(f"[INFO] Model: CLIP ViT-L-14 (zero-shot)")
    print(f"[INFO] Feature dimension: {query_embeddings.shape[1]}")
    print(f"[INFO] Number of queries: {len(query_files)}")
    print(f"[INFO] Gallery size: {len(gallery_files)}")
    print(f"[INFO] Top-K: {TOP_K}")
    print(f"[INFO] No training performed - pure zero-shot inference")
    print(f"[INFO] Results saved to: {SUBMISSION_PATH}")
    print(f"[INFO] === CLIP ZERO-SHOT EXPERIMENT COMPLETED! ===")

if __name__ == "__main__":
    main()