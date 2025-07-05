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

EMBED_DIM = 1024
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR_BASE = 1e-4
LR_BACKBONE = 1e-5

ARCFACE_SCALE = 64.0
ARCFACE_MARGIN = 0.5

UNFREEZE_LAYERS = 4
TOP_K = 10
SEED = 42

# ==== PATH ====
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "..")

TRAIN_DIR = os.path.join(BASE_DIR, "images_competition/train")
GALLERY_DIR = os.path.join(BASE_DIR, "images_competition/test", "gallery")
QUERY_DIR = os.path.join(BASE_DIR, "images_competition/test", "query")

MODEL_SAVE_PATH = os.path.join(script_dir, "clip_arcface_no_gem_trained.pt")
SUBMISSION_PATH = os.path.join(script_dir, "submission_arcface_no_gem2.py")

# ==== DEVICE & SEED ====
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

class CLIPArcFaceNoGeM(nn.Module):

    def __init__(self, clip_model, embed_dim=1024, num_classes=None, unfreeze_layers=4):
        super().__init__()
        self.clip = clip_model
        
        total_layers = len(clip_model.visual.transformer.resblocks)
        print(f"[INFO] Total transformer layers: {total_layers}, unfreezing last {unfreeze_layers}")
        
        for name, param in clip_model.visual.named_parameters():
            if 'resblocks' in name:
                try:
                    layer_idx = int(name.split('.')[2])
                    param.requires_grad = layer_idx >= total_layers - unfreeze_layers
                except:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        # Projection head
        clip_dim = 768
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        if num_classes:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
    
    def forward(self, x):
        clip_features = self.clip.visual(x)
        
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] CLIP features shape: {clip_features.shape}")
            self._debug_printed = True
        
        embeddings = F.normalize(self.projection(clip_features), dim=-1)
        
        if hasattr(self, 'classifier'):
            logits = self.classifier(embeddings)
            return embeddings, logits
        return embeddings
    
    def extract_features(self, x):
        with torch.no_grad():
            clip_feat = self.clip.visual(x)
            emb = F.normalize(self.projection(clip_feat), dim=-1)
        return emb

class ArcFaceLoss(nn.Module):
    def __init__(self, scale=64.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels, classifier_weights):
        normalized_weights = F.normalize(classifier_weights)
        
        cosine_sim = F.linear(embeddings, normalized_weights)
        
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        angle = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
        angle.add_(self.margin * one_hot)
        margin_cosine = torch.cos(angle)
        
        final_logits = margin_cosine * self.scale
        
        return self.ce_loss(final_logits, labels)

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

def train_arcface_no_gem(model, train_loader):
    print("[INFO] Starting training with ArcFace (NO GeM pooling)...")
    
    # Loss function
    arcface_loss_fn = ArcFaceLoss(scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.clip.visual.named_parameters() if p.requires_grad], 
         'lr': LR_BACKBONE, 'weight_decay': 1e-4},
        {'params': list(model.projection.parameters()) + list(model.classifier.parameters()), 
         'lr': LR_BASE, 'weight_decay': 1e-4}
    ])
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * (NUM_EPOCHS // 3), T_mult=1
    )
    
    scaler = GradScaler()
    
    model.train()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                embeddings, _ = model(images)
                
                arcface_loss = arcface_loss_fn(embeddings, labels, model.classifier.weight)
                
                total_batch_loss = arcface_loss
            
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += total_batch_loss.item()
            
            progress_bar.set_postfix({
                'ArcFace Loss': f'{arcface_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch}: ArcFace Loss = {avg_loss:.4f}')
        
    print("[INFO] Training completed!")

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
    print("[INFO] === ABLATION STUDY: CLIP + ArcFace (SENZA GeM) ===")
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

    num_classes = len(train_dataset.label_map)
    model = CLIPArcFaceNoGeM(
        clip_model, embed_dim=EMBED_DIM, num_classes=num_classes, unfreeze_layers=UNFREEZE_LAYERS
    ).to(device)
    
    print(f"[INFO] Training with {num_classes} classes, embed_dim={EMBED_DIM}")
    print(f"[INFO] ArcFace: scale={ARCFACE_SCALE}, margin={ARCFACE_MARGIN}")
    print(f"[INFO] NO GeM pooling - using standard CLIP class token pooling")
    
    # Training
    train_arcface_no_gem(model, train_loader)
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
    
    print("[INFO] Extracting embeddings...")
    query_embeddings, query_files = extract_embeddings(model, query_loader)
    gallery_embeddings, gallery_files = extract_embeddings(model, gallery_loader)
    
    print("[INFO] Performing retrieval...")
    results = cosine_retrieval(
        query_embeddings, gallery_embeddings, query_files, gallery_files, k=TOP_K
    )
    
    save_submission_dict(results, SUBMISSION_PATH)
    print(f"[INFO] Submission dictionary saved to {SUBMISSION_PATH}")
    print("[INFO] === CLIP + ArcFace (NO GeM) experiment completed! ===")

if __name__ == "__main__":
    main()