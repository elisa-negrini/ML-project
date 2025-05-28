import torch
from torchvision import datasets
from sklearn.neighbors import NearestNeighbors
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

MODEL_NAME = "facebook/dinov2-base"
try:
    dino_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    exit()

dino_model.eval()  # Solo inference

def get_feature_extractor(base_model, image_processor_func):
    def extractor(image_paths):
        embs = []
        for path in tqdm(image_paths, desc="Estrazione feature"):
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

def retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=20):
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Embedding vuoti. Retrieval non eseguibile.")
        return []

    model_nn = NearestNeighbors(n_neighbors=min(k, gallery_embs.shape[0]), metric='cosine')
    model_nn.fit(gallery_embs)
    distances, indices = model_nn.kneighbors(query_embs)

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
feature_extractor_fn = get_feature_extractor(dino_model, image_processor)
query_embs = feature_extractor_fn(query_files)
gallery_embs = feature_extractor_fn(gallery_files)

# === RETRIEVAL E SALVATAGGIO ===
if query_embs.shape[0] > 0 and gallery_embs.shape[0] > 0:
    submission_list = retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=10) # <- CAMBIARE QUESTO K
    data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }
    
    
    # submission(data, "Pretty Figure")


    submission_path = "submission/submission_dino_t7.py"
    save_submission_d(data, submission_path)

    
    # if you want json
    # submission_path = "submission/submission_dino_t7.json"
    # save_submission(submission, submission_path)
    print(f"✅ Submission salvata in: {submission_path}")
else:
    print("Embedding non estratti. Retrieval saltato.")
