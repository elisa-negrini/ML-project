import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== MODELLI ======
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

efficientnet_model = timm.create_model('tf_efficientnetv2_l', pretrained=True).to(device)
efficientnet_model.eval()

resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet_model.eval()

transform_eff = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
transform_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====== ESTRAZIONE EMBEDDING ======
def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def extract_clip_embeddings(image_paths):
    clip_model.eval()
    embeddings = []
    for path in tqdm(image_paths, desc="CLIP"):
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        embeddings.append(emb.cpu().numpy()[0])
    return l2_normalize(np.array(embeddings).astype("float32"))

def extract_dino_embeddings(image_paths):
    dino_model.eval()
    embeddings = []
    for path in tqdm(image_paths, desc="DINOv2"):
        image = Image.open(path).convert("RGB")
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = dino_model(**inputs)
            emb = output.last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu().numpy()[0])
    return l2_normalize(np.array(embeddings).astype("float32"))

def extract_efficientnet_embeddings(image_paths):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="EfficientNetV2"):
        batch_paths = image_paths[i:i+32]
        imgs = [transform_eff(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = efficientnet_model.forward_features(batch)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1)
            feats = feats.squeeze(-1).squeeze(-1)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

def extract_resnet_embeddings(image_paths):
    feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="ResNet50"):
        batch_paths = image_paths[i:i+32]
        imgs = [transform_resnet(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

# ====== MEDIA DEGLI EMBEDDING CON PCA ======
def mean_fuse_embeddings(*emb_lists, shared_pca=None):
    # Calculate the minimum dimension possible for PCA
    min_samples = min(emb.shape[0] for emb in emb_lists)
    # Choose a reasonable n_components that doesn't exceed the number of samples
    n_components = min(min_samples - 1, 64)  # Using min_samples - 1 to be safe
    
    reduced_embs = []
    for emb in emb_lists:
        # Only apply PCA if dimensions need to be reduced
        if emb.shape[1] > n_components:
            if shared_pca is not None:
                # Use pre-fitted PCA
                reduced = shared_pca.transform(emb)
            else:
                # Fit new PCA
                pca = PCA(n_components=n_components, whiten=True, random_state=42)
                reduced = pca.fit_transform(emb)
                if shared_pca is None:
                    shared_pca = pca
        else:
            # If original dimension is smaller, just use it as is
            reduced = emb
        reduced_embs.append(reduced)
    
    # Make sure all embeddings have the same dimension for stacking
    min_dim = min(emb.shape[1] for emb in reduced_embs)
    aligned_embs = [emb[:, :min_dim] for emb in reduced_embs]
    
    stacked = np.stack(aligned_embs, axis=0)
    fused = np.mean(stacked, axis=0)
    return l2_normalize(fused), shared_pca

# ====== RETRIEVAL ======
def retrieve_combined(query_embs, query_files, gallery_embs, gallery_files, k=50):
    model = NearestNeighbors(n_neighbors=k, metric='cosine')
    model.fit(gallery_embs)
    distances, indices = model.kneighbors(query_embs)
    results = []
    for i, query_path in enumerate(query_files):
        query_rel = query_path.replace("\\", "/")
        gallery_matches = [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
        results.append({
            "filename": query_rel,
            "gallery_images": gallery_matches
        })
    return results

def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # ====== MAIN ======
if __name__ == "__main__":
    query_folder = "testing_images7_fish/test/query"
    gallery_folder = "testing_images7_fish/test/gallery"

    query_files = sorted([os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    gallery_files = sorted([os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    print("🔍 Estrazione degli embedding...")

    query_clip = extract_clip_embeddings(query_files)
    gallery_clip = extract_clip_embeddings(gallery_files)

    query_dino = extract_dino_embeddings(query_files)
    gallery_dino = extract_dino_embeddings(gallery_files)

    query_eff = extract_efficientnet_embeddings(query_files)
    gallery_eff = extract_efficientnet_embeddings(gallery_files)

    query_resnet = extract_resnet_embeddings(query_files)
    gallery_resnet = extract_resnet_embeddings(gallery_files)

    print(f"Shape query embeddings:")
    print(f" - CLIP        : {query_clip.shape}")
    print(f" - DINOv2      : {query_dino.shape}")
    print(f" - EfficientNet: {query_eff.shape}")
    print(f" - ResNet50    : {query_resnet.shape}")

    # First create PCA transformers using all embeddings (query+gallery) to ensure consistent dimensions
    print("🔧 Creating PCA transformers...")
    
    # Combine query and gallery for PCA fitting
    all_clip = np.vstack([query_clip, gallery_clip])
    all_dino = np.vstack([query_dino, gallery_dino])
    all_eff = np.vstack([query_eff, gallery_eff])
    all_resnet = np.vstack([query_resnet, gallery_resnet])
    
    # PCA component limit based on number of samples and features
    min_samples = min(all_clip.shape[0], all_dino.shape[0], all_eff.shape[0], all_resnet.shape[0])
    n_components = min(min_samples - 1, 64)
    
    # Fit PCA on combined data
    pca_clip = PCA(n_components=n_components, whiten=True, random_state=42)
    pca_dino = PCA(n_components=n_components, whiten=True, random_state=42)
    pca_eff = PCA(n_components=n_components, whiten=True, random_state=42)
    pca_resnet = PCA(n_components=n_components, whiten=True, random_state=42)
    
    pca_clip.fit(all_clip)
    pca_dino.fit(all_dino)
    pca_eff.fit(all_eff)
    pca_resnet.fit(all_resnet)
    
    # Transform data with fitted PCAs
    query_clip_reduced = pca_clip.transform(query_clip)
    gallery_clip_reduced = pca_clip.transform(gallery_clip)
    
    query_dino_reduced = pca_dino.transform(query_dino)
    gallery_dino_reduced = pca_dino.transform(gallery_dino)
    
    query_eff_reduced = pca_eff.transform(query_eff)
    gallery_eff_reduced = pca_eff.transform(gallery_eff)
    
    query_resnet_reduced = pca_resnet.transform(query_resnet)
    gallery_resnet_reduced = pca_resnet.transform(gallery_resnet)
    
    # Stack and average
    query_stacked = np.stack([query_clip_reduced, query_dino_reduced, query_eff_reduced, query_resnet_reduced], axis=0)
    gallery_stacked = np.stack([gallery_clip_reduced, gallery_dino_reduced, gallery_eff_reduced, gallery_resnet_reduced], axis=0)
    
    query_fused = np.mean(query_stacked, axis=0)
    gallery_fused = np.mean(gallery_stacked, axis=0)
    
    # L2 normalize
    query_fused = l2_normalize(query_fused)
    gallery_fused = l2_normalize(gallery_fused)
    
    print(f"Shape fused embeddings:")
    print(f" - Query: {query_fused.shape}")
    print(f" - Gallery: {gallery_fused.shape}")

    print("📥 Retrieval in corso...")
    submission = retrieve_combined(query_fused, query_files, gallery_fused, gallery_files, k=50)

    save_submission(submission, "submission/ensemble_corrected.json")
    print("✅ Submission salvata.")