import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import faiss

# === CONFIGURAZIONE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
MODEL_NAME = "google/vit-base-patch16-224"

# === CARICAMENTO MODELLO E PROCESSOR ===
try:
    vit_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    exit()
vit_model.eval()

# === DEFINIZIONE MODELLO PER FINE-TUNING ===
class ImageClassifierFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(embed_dim, num_classes)
        if unfreeze_layers:
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]
        return self.classifier(features)

# === FUNZIONE COLLATE PERSONALIZZATA ===
def custom_collate(batch):
    tensors, labels = [], []
    for image, label in batch:
        tensors.append(image)
        labels.append(label)
    labels = default_collate(labels)
    return tensors, labels

# === FUNZIONE DI TRAINING ===
def train_model(model, dataloader, image_processor_func, epochs=10, lr=5e-5):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    base_params = [p for n, p in model.named_parameters() if "base_model" in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]
    optimizer_params = []
    if base_params:
        optimizer_params.append({'params': base_params, 'lr': lr / 10})
    if classifier_params:
        optimizer_params.append({'params': classifier_params, 'lr': lr})
    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for pil_images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = labels.to(device)
            try:
                inputs = image_processor_func(images=pil_images, return_tensors="pt", padding=True).to(device)
            except Exception as e:
                print(f"Errore durante il processing: {e}")
                continue
            optimizer.zero_grad()
            outputs = model(inputs.pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch+1} - Loss: {running_loss / len(dataloader):.4f}, Accuracy: {100.0 * correct / total:.2f}%")
    return model

# === FEATURE EXTRACTOR DAL MODELLO FINE-TUNED ===
def get_feature_extractor(trained_model, image_processor_func):
    model = trained_model.base_model
    model.eval()
    def extractor(image_paths):
        embs = []
        for path in tqdm(image_paths, desc="Extracting features"):
            try:
                image = Image.open(path).convert("RGB")
                inputs = image_processor_func(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(pixel_values=inputs.pixel_values)
                    emb = outputs.last_hidden_state[:, 0, :]
                    embs.append(emb.cpu().numpy()[0])
            except Exception as e:
                print(f"Errore con {path}: {e}")
        return np.array(embs).astype("float32") if embs else np.array([]).astype("float32")
    return extractor

# === RETRIEVAL CON FAISS ===
def retrieve_query_vs_gallery_faiss(query_embs, query_files, gallery_embs, gallery_files, k=10):
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Embedding vuoti. Retrieval non eseguibile.")
        return []
    faiss.normalize_L2(gallery_embs)
    faiss.normalize_L2(query_embs)
    index = faiss.IndexFlatIP(gallery_embs.shape[1])
    index.add(gallery_embs)
    distances, indices = index.search(query_embs, min(k, gallery_embs.shape[0]))
    results = []
    for i, query_path in enumerate(query_files):
        results.append({
            "filename": query_path.replace("\\", "/"),
            "gallery_images": [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
        })
    return results

# === SALVATAGGIO ===
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
#inserire qua def submission

# === ESECUZIONE ===
train_dataset = datasets.ImageFolder("testing_images8_animals/training")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
num_classes = len(train_dataset.classes)
print(f"Trovate {num_classes} classi: {train_dataset.classes}")

embed_dim = vit_model.config.hidden_size
model_to_train = ImageClassifierFineTuner(vit_model, embed_dim=embed_dim, num_classes=num_classes, unfreeze_layers=True)
trained_model = train_model(model_to_train, train_loader, image_processor, epochs=10, lr=1e-5)

feature_extractor_fn = get_feature_extractor(trained_model, image_processor)
query_files = [os.path.join("testing_images8_animals/test/query", f) for f in os.listdir("testing_images8_animals/test/query") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
gallery_files = [os.path.join("testing_images8_animals/test/gallery", f) for f in os.listdir("testing_images8_animals/test/gallery") if f.lower().endswith((".jpg", ".jpeg", ".png"))]

query_embs = feature_extractor_fn(query_files)
gallery_embs = feature_extractor_fn(gallery_files)

submission_list = retrieve_query_vs_gallery_faiss(query_embs, query_files, gallery_embs, gallery_files, k=10)
data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }

submission_path = "submission/submission_vit_faiss_t4.py"
save_submission_d(data, submission_path)

#submission(data, "Pretty Figure")
print(f"✅ Submission salvata in: {submission_path}")

