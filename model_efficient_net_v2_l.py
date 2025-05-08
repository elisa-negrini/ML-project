import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import timm
import numpy as np
import json
import os
from PIL import Image
from tqdm import tqdm

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform coerente con EfficientNet
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Modello EfficientNetV2-L preaddestrato
def get_feature_extractor():
    model = timm.create_model('tf_efficientnetv2_l', pretrained=True)
    model.eval()
    model = model.to(device)
    # Rimuovi classificatore finale
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    return feature_extractor

# Estrazione delle feature da una cartella
def extract_embeddings_from_folder(folder_path, model):
    image_paths = sorted([os.path.join(folder_path, fname)
                          for fname in os.listdir(folder_path)
                          if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

    all_embeddings = []
    filenames = []

    with torch.no_grad():
        for i in range(0, len(image_paths), 32):
            batch_paths = image_paths[i:i+32]
            imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            imgs = torch.stack(imgs).to(device)
            vecs = model(imgs).squeeze(-1).squeeze(-1)  # da (B, C, 1, 1) a (B, C)
            all_embeddings.append(vecs.cpu())
            filenames.extend(batch_paths)

    return torch.cat(all_embeddings, dim=0).numpy(), filenames

# Retrieval top-k tra query e galleria
def retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=5):
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

# Salva il file JSON finale
def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

# Main pipeline
if __name__ == "__main__":
    feature_extractor = get_feature_extractor()

    # Step 1: Estrai feature
    query_embeddings, query_files = extract_embeddings_from_folder("testing_images1/test/query", feature_extractor)
    gallery_embeddings, gallery_files = extract_embeddings_from_folder("testing_images1/test/gallery", feature_extractor)

    # Step 2: Retrieval
    submission = retrieve_query_vs_gallery(query_embeddings, query_files, gallery_embeddings, gallery_files, k=30)

    # Step 3: Salva submission
    submission_path = "submission/submission.json"
    save_submission(submission, submission_path)
    print(f"✅ Submission salvata in: {submission_path}")
