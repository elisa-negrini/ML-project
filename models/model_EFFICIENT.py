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

# Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform coerente con EfficientNetV2
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# 1. Creazione modello pretrained senza modifiche
def get_model():
    model = timm.create_model('tf_efficientnetv2_l', pretrained=True)
    model.eval()
    return model.to(device)

# 2. Estrazione feature usando forward_features
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

            vecs = model.forward_features(imgs)
            vecs = torch.nn.functional.adaptive_avg_pool2d(vecs, 1)
            vecs = vecs.squeeze(-1).squeeze(-1)

            all_embeddings.append(vecs.cpu())
            filenames.extend(batch_paths)

    return torch.cat(all_embeddings, dim=0).numpy(), filenames

# 3. Retrieval top-k tra query e gallery
def retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=5):
    nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn_model.fit(gallery_embs)
    distances, indices = nn_model.kneighbors(query_embs)

    results = []
    for i, query_path in enumerate(query_files):
        query_rel = query_path.replace("\\", "/")
        gallery_matches = [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
        results.append({
            "filename": query_rel,
            "gallery_images": gallery_matches
        })
    return results

# 4. Salva il file JSON di submission
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


# MAIN PIPELINE
if __name__ == "__main__":
    model = get_model()

    query_embeddings, query_files = extract_embeddings_from_folder("testing_images7_fish/test/query", model)
    gallery_embeddings, gallery_files = extract_embeddings_from_folder("testing_images7_fish/test/gallery", model)

    submission = retrieve_query_vs_gallery(query_embeddings, query_files, gallery_embeddings, gallery_files, k=50) # <- CAMBIA QUESTO K

    data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission
    }
    submission_path = "submission/submission_efficient_t7.py"
    save_submission_d(data, submission_path)

    
    # if you want json
    # submission_path = "submission/submission_efficient_t7.json"
    # save_submission(submission, submission_path)
    print(f"✅ Submission salvata in: {submission_path}")
