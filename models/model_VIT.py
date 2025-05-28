import torch
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "google/vit-base-patch16-224"
try:
    vit_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    exit()

vit_model.eval()  # Solo inference

def get_feature_extractor(base_model, image_processor_func):
    def extractor(image_paths):
        embs = []
        for path in tqdm(image_paths, desc="Estrazione feature (ViT)"):
            try:
                image = Image.open(path).convert("RGB")
                inputs = image_processor_func(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = base_model(pixel_values=inputs.pixel_values)
                    emb = outputs.last_hidden_state[:, 0, :]  # CLS token
                    embs.append(emb.cpu().numpy()[0])
            except Exception as e:
                print(f"Errore con {path}: {e}")
                continue
        return np.array(embs).astype("float32") if embs else np.array([]).astype("float32")
    return extractor

def retrieve_query_vs_gallery_faiss(query_embs, query_files, gallery_embs, gallery_files, k=10):
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Embedding vuoti. Retrieval non eseguibile.")
        return []

    # Normalizzazione L2 (per cosine similarity)
    faiss.normalize_L2(gallery_embs)
    faiss.normalize_L2(query_embs)

    dim = gallery_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = inner product = cosine sim (dopo L2 norm)
    index.add(gallery_embs)

    distances, indices = index.search(query_embs, min(k, gallery_embs.shape[0]))

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

def save_submission_d(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            f.write(f'    "{key}": {value},\n')
        f.write("}\n")

#da aggiungere qua def submission

# === CARICAMENTO FILE QUERY E GALLERY ===
query_folder = "testing_images7_fish/test/query"
gallery_folder = "testing_images7_fish/test/gallery"

try:
    query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
except Exception as e:
    print(f"Errore nel caricamento immagini: {e}")
    exit()

if not query_files or not gallery_files:
    print("Query o gallery vuote.")
    exit()

print(f"Trovate {len(query_files)} query e {len(gallery_files)} gallery.")

# === ESTRAZIONE FEATURE ===
feature_extractor_fn = get_feature_extractor(vit_model, image_processor)
query_embs = feature_extractor_fn(query_files)
gallery_embs = feature_extractor_fn(gallery_files)

# === RETRIEVAL E SALVATAGGIO ===
if query_embs.shape[0] > 0 and gallery_embs.shape[0] > 0:
    submission_list = retrieve_query_vs_gallery_faiss(query_embs, query_files, gallery_embs, gallery_files, k=10)  # <- CAMBIA QUA IL K
    data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }
    submission_path = "submission/submission_vit_faiss_t7.py"
    save_submission_d(data, submission_path)

    #submission(data, "Pretty Figure")
    print(f"✅ Submission salvata in: {submission_path}")
else:
    print("Embedding non estratti. Retrieval saltato.")
