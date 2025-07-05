import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
import os
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_feature_extractor():
    model = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor.to(device)

def extract_embeddings_from_folder(folder_path, model, batch_size=32):
    image_paths = sorted([os.path.join(folder_path, fname)
                          for fname in os.listdir(folder_path)
                          if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

    all_embeddings = []
    filenames = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            imgs = torch.stack(imgs).to(device)
            vecs = model(imgs).squeeze(-1).squeeze(-1)
            all_embeddings.append(vecs.cpu())
            filenames.extend(batch_paths)

    return torch.cat(all_embeddings, dim=0).numpy(), filenames

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

# ---- MAIN FLOW ----

feature_extractor = get_feature_extractor()

query_embeddings, query_files = extract_embeddings_from_folder("images_competition/test/query", feature_extractor)
gallery_embeddings, gallery_files = extract_embeddings_from_folder("images_competition/test/gallery", feature_extractor)

submission_list = retrieve_query_vs_gallery(query_embeddings, query_files, gallery_embeddings, gallery_files, k=10)

data = {
    os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
    for entry in submission_list
}

submission_path = "submission/submission_resnet50_t6.py"
save_submission_d(data, submission_path)

print(f"✅ Submission salvata in: {submission_path}")