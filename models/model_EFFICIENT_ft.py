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

# 1. Creazione modello con classificatore adattato
def get_model(num_classes):
    model = timm.create_model('tf_efficientnetv2_l', pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model.to(device)

# 2. Fine-tuning del modello sul training set
def train_model(model, dataloader, epochs=5, lr=1e-4):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Loss: {running_loss/len(dataloader):.4f}")
    return model

# 3. Estrazione feature usando forward_features
def extract_embeddings_from_folder(folder_path, model):
    image_paths = sorted([os.path.join(folder_path, fname)
                          for fname in os.listdir(folder_path)
                          if fname.lower().endswith(('.jpg', '.jpeg', '.png'))])

    all_embeddings = []
    filenames = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), 32):
            batch_paths = image_paths[i:i+32]
            imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            imgs = torch.stack(imgs).to(device)

            # FIX: aggiunto pooling globale per passare da 4D a 2D
            vecs = model.forward_features(imgs)
            vecs = torch.nn.functional.adaptive_avg_pool2d(vecs, 1)
            vecs = vecs.squeeze(-1).squeeze(-1)

            all_embeddings.append(vecs.cpu())
            filenames.extend(batch_paths)

    return torch.cat(all_embeddings, dim=0).numpy(), filenames


# 4. Retrieval top-k tra query e galleria
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

# 5. Salva il file JSON di submission
def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

# MAIN PIPELINE
if __name__ == "__main__":
    # Step 1: Fine-tuning sul training set
    train_dataset = datasets.ImageFolder("testing_images5/training", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = get_model(num_classes=len(train_dataset.classes))
    #model = train_model(model, train_loader, epochs=5)

    # Step 2: Estrai embedding da query e gallery
    query_embeddings, query_files = extract_embeddings_from_folder("testing_images5/test/query", model)
    gallery_embeddings, gallery_files = extract_embeddings_from_folder("testing_images5/test/gallery", model)

    # Step 3: Retrieval
    submission = retrieve_query_vs_gallery(query_embeddings, query_files, gallery_embeddings, gallery_files, k=50) # <- CAMBIA QUESTO K

    # Step 4: Salva submission
    submission_path = "submission/submission_efficient_ft_t5.json"
    save_submission(submission, submission_path)
    print(f"✅ Submission salvata in: {submission_path}")
