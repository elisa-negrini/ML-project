import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
import timm

# ====== CONFIGURAZIONE ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== CARICAMENTO MODELLI PRE-ADDESTRATI ======

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_clip_embeddings(image_paths):
    clip_model.eval()
    embeddings = []
    for path in tqdm(image_paths, desc="CLIP"):
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        embeddings.append(emb.cpu().numpy()[0])
    return np.array(embeddings).astype("float32")

# DINOv2
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

def extract_dino_embeddings(image_paths):
    dino_model.eval()
    embeddings = []
    for path in tqdm(image_paths, desc="DINOv2"):
        image = Image.open(path).convert("RGB")
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = dino_model(**inputs)
            emb = output.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings.append(emb.cpu().numpy()[0])
    return np.array(embeddings).astype("float32")

# EfficientNetV2
transform_eff = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

efficientnet_model = timm.create_model('tf_efficientnetv2_l', pretrained=True).to(device)
efficientnet_model.eval()

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
    return torch.cat(embeddings, dim=0).numpy().astype("float32")

# ResNet50
resnet_model = models.resnet50(pretrained=True).to(device)
resnet_model.eval()
transform_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_resnet_embeddings(image_paths):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="ResNet50"):
        batch_paths = image_paths[i:i+32]
        imgs = [transform_resnet(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = resnet_model.forward(batch)
        embeddings.append(feats.cpu())
    return torch.cat(embeddings, dim=0).numpy().astype("float32")

# ====== COMBINAZIONE E RETRIEVAL ======

def concatenate_embeddings(*emb_lists):
    return np.concatenate(emb_lists, axis=1)

def retrieve_combined(query_embs, query_files, gallery_embs, gallery_files, k=30):
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
    print("let's start")
    query_folder = "testing_images7_fish/test/query"
    gallery_folder = "testing_images7_fish/test/gallery"

    query_files = sorted([os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    gallery_files = sorted([os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # Estrazione embedding da ciascun modello
    query_clip = extract_clip_embeddings(query_files)
    gallery_clip = extract_clip_embeddings(gallery_files)

    query_dino = extract_dino_embeddings(query_files)
    gallery_dino = extract_dino_embeddings(gallery_files)

    query_eff = extract_efficientnet_embeddings(query_files)
    gallery_eff = extract_efficientnet_embeddings(gallery_files)

    query_resnet = extract_resnet_embeddings(query_files)
    gallery_resnet = extract_resnet_embeddings(gallery_files)

    # Concatenazione
    query_total = concatenate_embeddings(query_clip, query_dino, query_eff, query_resnet)
    gallery_total = concatenate_embeddings(gallery_clip, gallery_dino, gallery_eff, gallery_resnet)

    # Retrieval
    submission = retrieve_combined(query_total, query_files, gallery_total, gallery_files, k=50)

    # Salvataggio
    save_submission(submission, "submission/ensemble_CDER.json")
    print("✅ Submission salvata in: submission/ensemble_submission.json")
