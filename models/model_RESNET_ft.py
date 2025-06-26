import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
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

def get_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, dataloader, epochs=5, lr=1e-4):
    model = model.to(device)
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

def get_feature_extractor(trained_model):
    feature_extractor = nn.Sequential(*list(trained_model.children())[:-1])
    feature_extractor.eval()
    return feature_extractor.to(device)

def retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=10):
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

def save_submission_d(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            f.write(f'    "{key}": {value},\n')
        f.write("}\n")



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
            vecs = model(imgs).squeeze(-1).squeeze(-1)
            all_embeddings.append(vecs.cpu())
            filenames.extend(batch_paths)

    return torch.cat(all_embeddings, dim=0).numpy(), filenames

# Step 1: Fine-tune il modello sul training set
train_dataset = datasets.ImageFolder("train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = get_model(num_classes=len(train_dataset.classes))
model = train_model(model, train_loader, epochs=5)

# Step 2: Estrai features da query e gallery
feature_extractor = get_feature_extractor(model)
query_embeddings, query_files = extract_embeddings_from_folder("test/query", feature_extractor)
gallery_embeddings, gallery_files = extract_embeddings_from_folder("test/gallery", feature_extractor)

# Step 3: Retrieval
submission_list = retrieve_query_vs_gallery(query_embeddings, query_files, gallery_embeddings, gallery_files, k=10) # <- CAMBIARE QUESTO K 

data = {
    os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
    for entry in submission_list
}

# submission(data, "Pretty Figure")

# Step 4: Salvataggio nella repo
submission_path = "submission/submission_resnet50_ft.py"

save_submission_d(data, submission_path)


# if you want json
# submission_path = "submission/submission_resnet50_ft_t5.json"
# save_submission(submission, submission_path)
print(f"✅ Submission salvata in: {submission_path}")

