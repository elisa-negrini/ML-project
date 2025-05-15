import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import json
import faiss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica modello e processor pre-addestrato
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_features(image_paths):
    clip_model.eval()
    embs = []
    for path in tqdm(image_paths, desc="Extracting features"):
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        embs.append(emb.cpu().numpy()[0])
    return np.array(embs).astype("float32")

def retrieve(query_embs, gallery_embs, query_files, gallery_files, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")

    # matric = "cosine", "euclidean", "manhattan"
    # CLIP funziona bene con la distanza coseno perché gli embeddings sono spesso normalizzati internamente.

    # algorithm = "auto" (default), "ball_tree", "kd_tree", "brute", "faiss"

    nn.fit(gallery_embs)
    distances, indices = nn.kneighbors(query_embs)

    results = []
    for i, query_path in enumerate(query_files):
        gallery_matches = [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
        results.append({
            "filename": query_path.replace("\\", "/"),
            "gallery_images": gallery_matches
        })
    return results

# def retrieve_with_faiss(query_embs, gallery_embs, query_files, gallery_files, k=5):
#     faiss.normalize_L2(query_embs)
#     faiss.normalize_L2(gallery_embs)
#     index = faiss.IndexFlatIP(gallery_embs.shape[1])  # 512 for CLIP
#     index.add(gallery_embs)
#     distances, indices = index.search(query_embs, k)

#     results = []
#     for i, query_path in enumerate(query_files):
#         matches = [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
#         results.append({
#             "filename": query_path.replace("\\", "/"),
#             "gallery_images": matches
#         })
#     return results


# Percorsi cartelle immagini
query_folder = "testing_images6_clothes/test/query"
gallery_folder = "testing_images6_clothes/test/gallery"

# Lista file immagini
query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.endswith(".jpg")]
gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.endswith(".jpg")]

# Estrai feature
query_embs = extract_features(query_files)
gallery_embs = extract_features(gallery_files)

# Recupero immagini simili
submission = retrieve(query_embs, gallery_embs, query_files, gallery_files, k=49)

# Salva risultati
output_path = "submission/submission_clip_t6.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(submission, f, indent=2)

print(f"✅ Submission salvata in: {output_path}")
