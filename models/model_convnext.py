import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import json
from tqdm import tqdm
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica ConvNeXt preaddestrato
weights = ConvNeXt_Base_Weights.DEFAULT
convnext_model = convnext_base(weights=weights).to(device)
convnext_model.eval()

# Definisci il preprocessing per ConvNeXt
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # ConvNeXt si aspetta immagini 224x224
    transforms.ToTensor(),
    weights.transforms(),  # Utilizza direttamente le trasformazioni fornite dai pesi
])

# Funzione per estrarre le feature da una lista di immagini
def extract_features(image_paths):
    features = []
    for path in tqdm(image_paths, desc="Extracting features"):
        image = Image.open(path).convert("RGB")
        inputs = preprocess(image).unsqueeze(0).to(device)  # Aggiungi batch dimension
        with torch.no_grad():
            feature = convnext_model(inputs)
        features.append(feature.cpu().numpy()[0])  # Estrai feature come array numpy
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

def save_submission_d(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            f.write(f'    "{key}": {value},\n')
        f.write("}\n")

# ===============================
# ESECUZIONE
# ===============================
query_folder = "testing_images5/test/query"
gallery_folder = "testing_images5/test/gallery"

query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.endswith(".jpg")]
gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.endswith(".jpg")]

query_embs = extract_features(query_files)
gallery_embs = extract_features(gallery_files)

submission_list = retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=10)

data = {
    os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
    for entry in submission_list
}
print(data)
submission_path = "submission/submission_convnext_t6.py"
save_submission_d(data, submission_path)
#submission(data, "Pretty Figure")

# if you want json
# submission_path = "submission/submission_convnext_t6.json"
# save_submission(submission, submission_path)
print(f"✅ Submission salvata in: {submission_path}")