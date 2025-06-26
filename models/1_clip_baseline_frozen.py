import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
import open_clip
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random

# ==== CONFIGURAZIONE IPERPARAMETRI ====
EMBED_DIM = 1024
NUM_EPOCHS = 15
BATCH_SIZE = 32
LR_BASE = 1e-4

TOP_K = 10
SEED = 42

# ==== PATH ====
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "..")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
GALLERY_DIR = os.path.join(BASE_DIR, "test", "gallery")
QUERY_DIR = os.path.join(BASE_DIR, "test", "query")

MODEL_SAVE_PATH = os.path.join(script_dir, "clip_baseline_frozen.pt")
SUBMISSION_PATH = os.path.join(script_dir, "submission_baseline_frozen.py")
LOGS_PATH = os.path.join(script_dir, "logs_baseline_frozen.json")

# ==== DEVICE & SEED ====
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

# ==== ARCHITETTURA: CLIP BASELINE (FROZEN) ====

class CLIPBaseline(nn.Module):
    """
    CLIP Baseline con backbone completamente freezato.
    Solo classificatore lineare trainabile.
    """
    def __init__(self, clip_model, embed_dim=1024, num_classes=None):
        super().__init__()
        self.clip = clip_model
        
        # Freeze tutto il backbone CLIP
        for param in clip_model.parameters():
            param.requires_grad = False
        
        # Semplice projection head
        clip_dim = 768  # ViT-L-14 output dim
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Classificatore lineare
        if num_classes:
            self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # Estrai features con CLIP (frozen)
        with torch.no_grad():
            features = self.clip.encode_image(x)  # [B, 1024]
        
        # Proiezione trainabile
        embeddings = self.projection(features.float())
        
        if hasattr(self, 'classifier'):
            logits = self.classifier(embeddings)
            return embeddings, logits
        return embeddings
    
    def extract_features(self, x):
        """Metodo per estrazione features durante inference"""
        with torch.no_grad():
            clip_features = self.clip.encode_image(x)
            embeddings = self.projection(clip_features.float())
            return F.normalize(embeddings, dim=-1)

# ==== DATASET ====
class ImprovedIdentityDataset(Dataset):
    def __init__(self, root_dir, preprocess_fn, is_training=True):
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.preprocess_fn = preprocess_fn
        self.is_training = is_training
        
        if is_training:
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
            self.post_process = transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        else:
            self.augmentation = transforms.Compose([transforms.Resize((224, 224))])
            self.post_process = None
        
        idx = 0
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"[ERROR] Directory not found: {root_dir}")
        for identity in sorted(os.listdir(root_dir)):
            identity_path = os.path.join(root_dir, identity)
            if not os.path.isdir(identity_path): continue
            if identity not in self.label_map:
                self.label_map[identity] = idx
                idx += 1
            label = self.label_map[identity]
            for img_file in os.listdir(identity_path):
                if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.samples.append(os.path.join(identity_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        try:
            image = default_loader(path).convert("RGB")
            image = self.augmentation(image)
            if self.preprocess_fn: image = self.preprocess_fn(image)
            if self.post_process: image = self.post_process(image)
            return image, label
        except Exception as e:
            print(f"[ERROR] Loading {path}: {e}")
            return torch.zeros(3, 224, 224), label

class FlatTestDataset(Dataset):
    def __init__(self, root_dir, preprocess_fn):
        self.root_dir = root_dir
        self.preprocess_fn = preprocess_fn
        self.samples = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"[ERROR] Directory not found: {root_dir}")
        for img_file in os.listdir(root_dir):
            if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                self.samples.append(os.path.join(root_dir, img_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.preprocess_fn: image = self.preprocess_fn(image)
            return image, img_path
        except Exception as e:
            print(f"[ERROR] Loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), img_path

# ==== TRAINING BASELINE ====

def train_baseline(model, train_loader):
    """
    Training baseline con solo CrossEntropy loss.
    """
    print("[INFO] Starting baseline training (frozen backbone)...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=LR_BASE, weight_decay=1e-4
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * (NUM_EPOCHS // 3), T_mult=1
    )
    
    scaler = GradScaler()
    
    model.train()
    epoch_logs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                embeddings, logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        epoch_logs.append({
            'epoch': epoch,
            'loss': avg_loss,
            'train_acc': train_acc
        })
        
        print(f'Epoch {epoch}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}%')
    
    # Salva logs
    with open(LOGS_PATH, 'w') as f:
        json.dump(epoch_logs, f, indent=2)
    
    print("[INFO] Training completed!")

# ==== ESTRAZIONE EMBEDDING E SUBMISSION ====

def extract_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    file_paths = []
    with torch.no_grad():
        for images, paths_batch in tqdm(data_loader, desc='Extracting embeddings'):
            images = images.to(device)
            emb = model.extract_features(images)
            embeddings.append(emb.cpu())
            file_paths.extend(paths_batch)
    if not embeddings:
        return torch.tensor([]), []
    return torch.vstack(embeddings), file_paths

def save_submission_dict(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            formatted_key = os.path.basename(key)
            formatted_values = [os.path.basename(v) for v in value]
            f.write(f'    "{formatted_key}": {json.dumps(formatted_values)},\n')
        f.write("}\n")

def cosine_retrieval(query_embeddings, gallery_embeddings, query_files, gallery_files, k=10):
    similarities = query_embeddings.numpy() @ gallery_embeddings.numpy().T
    ranks = np.argsort(similarities, axis=1)[:, ::-1]
    results = {}
    for i, query_file in enumerate(query_files):
        top_k_indices = ranks[i][:k]
        matches = [gallery_files[idx] for idx in top_k_indices]
        results[query_file] = matches
    return results

# ==== MAIN PIPELINE ====
def main():
    print("[INFO] === ABLATION STUDY: CLIP BASELINE (FROZEN) ===")
    print("[INFO] Loading ViT-L-14 model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model.to(device)
    
    print("[INFO] Creating datasets...")
    try:
        train_dataset = ImprovedIdentityDataset(TRAIN_DIR, preprocess, is_training=True)
        if len(train_dataset) == 0:
            print(f"[CRITICAL ERROR] No training samples found in {TRAIN_DIR}. Exiting.")
            return

        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            sampler=MPerClassSampler(train_dataset.labels, m=4, length_before_new_iter=len(train_dataset)),
            num_workers=4
        )
        
        query_dataset = FlatTestDataset(QUERY_DIR, preprocess)
        gallery_dataset = FlatTestDataset(GALLERY_DIR, preprocess)

        if len(query_dataset) == 0 or len(gallery_dataset) == 0:
            print("[CRITICAL ERROR] Query or Gallery dataset is empty. Exiting.")
            return

        query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    except FileNotFoundError as e:
        print(e)
        return

    # Inizializza modello baseline
    num_classes = len(train_dataset.label_map)
    model = CLIPBaseline(
        clip_model, embed_dim=EMBED_DIM, num_classes=num_classes
    ).to(device)
    
    print(f"[INFO] Training with {num_classes} classes, embed_dim={EMBED_DIM}")
    print(f"[INFO] Backbone frozen: {not any(p.requires_grad for p in model.clip.parameters())}")
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training
    train_baseline(model, train_loader)
    
    # Salva modello
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
    
    # Estrai embeddings
    print("[INFO] Extracting embeddings...")
    query_embeddings, query_files = extract_embeddings(model, query_loader)
    gallery_embeddings, gallery_files = extract_embeddings(model, gallery_loader)
    
    # Retrieval
    print("[INFO] Performing retrieval...")
    results = cosine_retrieval(
        query_embeddings, gallery_embeddings, query_files, gallery_files, k=TOP_K
    )
    
    # Salva submission
    save_submission_dict(results, SUBMISSION_PATH)
    print(f"[INFO] Submission dictionary saved to {SUBMISSION_PATH}")
    print(f"[INFO] Training logs saved to {LOGS_PATH}")

if __name__ == "__main__":
    main()