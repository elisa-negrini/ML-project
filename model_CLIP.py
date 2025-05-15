import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Usiamo direttamente le trasformazioni di CLIP
def get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Non applichiamo normalizzazione qui, lasciamo che clip_processor la gestisca
    ])

# Classe per il fine-tuning che permette di aggiornare tutti i parametri
class CLIPFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Se vogliamo fare fine-tuning, sblocchiamo i parametri del modello base
        if unfreeze_layers:
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = True
        else:
            # Altrimenti, congela il modello base (solo feature extraction)
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        # Non usiamo torch.no_grad() qui per permettere il backpropagation
        features = self.base_model.get_image_features(pixel_values=pixel_values)
        return self.classifier(features)

def train_model(model, dataloader, epochs=5, lr=5e-5):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Creiamo due gruppi di parametri con learning rate diversi
    base_params = [p for n, p in model.named_parameters() if "base_model" in n]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': lr / 10},  # LR più basso per il modello pre-addestrato
        {'params': classifier_params, 'lr': lr}   # LR standard per il classificatore
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            # Preprocessing con clip_processor, specificiamo do_rescale=False
            # poiché le immagini sono già in formato [0,1] da transforms.ToTensor()
            inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)

            optimizer.zero_grad()
            outputs = model(inputs.pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Calcolo dell'accuratezza
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return model

def get_feature_extractor(model):
    def extractor(image_paths):
        model.eval()  # Assicuriamoci che il modello sia in modalità valutazione
        embs = []
        for path in tqdm(image_paths, desc="Extracting features"):
            image = Image.open(path).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                # Usiamo direttamente il modello fine-tuned per estrarre features
                emb = model.base_model.get_image_features(**inputs)
            embs.append(emb.cpu().numpy()[0])
        return np.array(embs).astype("float32")
    return extractor

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

def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

# ===============================
# ESECUZIONE COMPLETA
# ===============================

# Step 1: Prepara il dataset con le trasformazioni corrette
transform = get_transform()
#train_dataset = datasets.ImageFolder("testing_images4/training", transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Batch size ridotto per evitare OOM

# Step 2: Fine-tune CLIP
model = CLIPFineTuner(clip_model, embed_dim=512, num_classes=23)
#model = train_model(model, train_loader, epochs=10, lr=1e-4)  # Aumentato numero di epoche

# Step 3: Estrai features da query e gallery usando il modello fine-tuned
extractor = get_feature_extractor(model)

query_folder = "ML-project/testing_images7_fish/test/query"
gallery_folder = "ML-project/testing_images7_fish/test/gallery"
query_files = [os.path.join(query_folder, fname) for fname in os.listdir(query_folder) if fname.endswith(".jpg")]
gallery_files = [os.path.join(gallery_folder, fname) for fname in os.listdir(gallery_folder) if fname.endswith(".jpg")]

query_embs = extractor(query_files)
gallery_embs = extractor(gallery_files)

# Step 4: Retrieval
submission = retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=49)

# Step 5: Salvataggio
submission_path = "ML-project//submission/submission_clip_t7.json"
save_submission(submission, submission_path)
print(f"✅ Submission salvata in: {submission_path}")