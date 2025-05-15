import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica CLIP preaddestrato
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Funzione per estrarre le feature da una lista di immagini
def extract_features(image_paths):
    clip_model.eval()
    features = []
    for path in tqdm(image_paths, desc="Extracting features"):
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        features.append(image_features.cpu().numpy()[0])
    return np.array(features).astype("float32")

# Retrieval k-NN
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

# Salvataggio JSON
def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

# ===============================
# ESECUZIONE
# ===============================

query_folder = "testing_images6_clothes/query"
gallery_folder = "testing_images6_clothes/gallery"

query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.endswith(".jpg")]
gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.endswith(".jpg")]

query_embs = extract_features(query_files)
gallery_embs = extract_features(gallery_files)

submission = retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=200)

submission_path = "testing_images6_clothes/submission/submission.json"
save_submission(submission, submission_path)
print(f"✅ Submission salvata in: {submission_path}")
