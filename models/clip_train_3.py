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
import math

# ==== CONFIGURAZIONE IPERPARAMETRI ====
EMBED_DIM = 1024  # Ridotto da 2048 per efficienza
NUM_EPOCHS = 15   # Leggermente aumentato per compensare architettura più complessa
BATCH_SIZE = 32
LR_BASE = 1e-4
LR_BACKBONE = 1e-5

# -- Parametri ArcFace --
ARCFACE_SCALE = 64.0
ARCFACE_MARGIN = 0.5

# -- Parametri Center Loss --
CENTER_LOSS_WEIGHT = 0.001
CENTER_LR = 0.5

# -- Parametri Mixup --
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5  # Probabilità di applicare mixup

UNFREEZE_LAYERS = 4
TOP_K = 10
SEED = 42

# ==== PATH ====
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, "..")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
GALLERY_DIR = os.path.join(BASE_DIR, "test", "gallery")
QUERY_DIR = os.path.join(BASE_DIR, "test", "query")

MODEL_SAVE_PATH = os.path.join(script_dir, "clip_gem_arcface_trained.pt")
SUBMISSION_PATH = os.path.join(script_dir, "submission_gem_arcface.py")

# ==== DEVICE & SEED ====
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

# ==== ARCHITETTURA: CLIP + GeM + ArcFace ====

class GeM(nn.Module):
    """
    Generalized Mean Pooling per aggregare patch features in modo più sofisticato.
    Il parametro p è learnable e controlla il focus su feature dominanti.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        # x: [B, N_patches, dim]
        # Calcola la media generalizzata: (mean(x^p))^(1/p)
        return F.avg_pool1d(
            x.clamp(min=self.eps).pow(self.p).transpose(1, 2), 
            kernel_size=x.size(1)
        ).pow(1.0 / self.p).squeeze(-1)

class CLIPGeMArcFace(nn.Module):
    """
    Architettura: CLIP + GeM Pooling + ArcFace
    - Estrae patch features dal ViT (senza class token)
    - Applica GeM pooling per aggregazione sofisticata
    - Proiezione finale con normalizzazione per ArcFace
    """
    def __init__(self, clip_model, embed_dim=1024, num_classes=None, unfreeze_layers=4):
        super().__init__()
        self.clip = clip_model
        
        # Freeze/unfreeze strategia invariata
        total_layers = len(clip_model.visual.transformer.resblocks)
        for name, param in clip_model.visual.named_parameters():
            if 'resblocks' in name:
                try:
                    layer_idx = int(name.split('.')[2])
                    param.requires_grad = layer_idx >= total_layers - unfreeze_layers
                except:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
        # GeM Pooling con parametro learnable
        self.gem_pooling = GeM(p=3.0)
        
        # Projection head più semplice ma efficace
        clip_dim = 1024  # For ViT-L-14, this is typically 768, but your debug output shows 1024.
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),  # 768 -> 1024
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # ArcFace classifier (senza bias come da best practice)
        if num_classes:
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
    
    def forward(self, x):
        # Estrai patch features (senza class token)
        patch_features = self.extract_patch_features(x)  # [B, N_patches, dim]
        
        # Debug dimensioni
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] Patch features shape: {patch_features.shape}")
            self._debug_printed = True
        
        # Applica GeM pooling
        pooled_features = self.gem_pooling(patch_features)  # [B, dim]
        
        # Debug dimensioni dopo pooling
        if hasattr(self, '_debug_printed') and not hasattr(self, '_debug_pooled'):
            print(f"[DEBUG] Pooled features shape: {pooled_features.shape}")
            self._debug_pooled = True
        
        # Proiezione e normalizzazione per ArcFace
        embeddings = F.normalize(self.projection(pooled_features), dim=-1)
        
        if hasattr(self, 'classifier'):
            logits = self.classifier(embeddings)
            return embeddings, logits
        return embeddings
    
    def extract_patch_features(self, x):
        """
        Estrae features intermediate dal ViT prima del final pooling.
        Rimuove il class token per mantenere solo le patch features.
        """
        # Preprocessing ViT standard
        x = self.clip.visual.conv1(x)  # Patch embedding
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, N_patches, dim]
        
        # Aggiungi class token
        class_token = self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)  # [B, 1+N_patches, dim]
        
        # Positional embedding e layer norm
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        
        # Passa attraverso i transformer blocks
        for layer in self.clip.visual.transformer.resblocks:
            x = layer(x)
        
        # Rimuovi class token, mantieni solo patch tokens
        return x[:, 1:]  # [B, N_patches, dim]
    
    def extract_features(self, x):
        """Metodo per estrazione features durante inference"""
        with torch.no_grad():
            patch_feat = self.extract_patch_features(x)
            pooled = self.gem_pooling(patch_feat)
            emb = F.normalize(self.projection(pooled), dim=-1)
        return emb

# ==== CENTER LOSS ====

class CenterLoss(nn.Module):
    """
    Center Loss per rendere gli embeddings più compatti.
    Mantiene un centro learnable per ogni classe e penalizza la distanza.
    """
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        # Calcola distanza squared L2 tra embeddings e centri corrispondenti
        centers_batch = self.centers[labels]  # [B, feat_dim]
        return torch.sum((embeddings - centers_batch) ** 2) / (2.0 * batch_size)

# ==== MIXUP AUGMENTATION ====

def mixup_data(x, y, alpha=0.2):
    """
    Implementa Mixup per migliorare la generalizzazione real→synthetic.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calcola loss per dati mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==== ARCFACE LOSS (invariata ma con supporto mixup) ====

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
    
    def forward_mixup(self, embeddings, y_a, y_b, lam, classifier_weights):
        """Versione per mixup che calcola loss per entrambe le label"""
        normalized_weights = F.normalize(classifier_weights)
        cosine_sim = F.linear(embeddings, normalized_weights)
        
        # Per y_a
        one_hot_a = torch.zeros_like(cosine_sim)
        one_hot_a.scatter_(1, y_a.view(-1, 1).long(), 1)
        angle_a = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
        angle_a.add_(self.margin * one_hot_a)
        margin_cosine_a = torch.cos(angle_a)
        final_logits_a = margin_cosine_a * self.scale
        
        # Per y_b
        one_hot_b = torch.zeros_like(cosine_sim)
        one_hot_b.scatter_(1, y_b.view(-1, 1).long(), 1)
        angle_b = torch.acos(torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7))
        angle_b.add_(self.margin * one_hot_b)
        margin_cosine_b = torch.cos(angle_b)
        final_logits_b = margin_cosine_b * self.scale
        
        loss_a = self.ce_loss(final_logits_a, y_a)
        loss_b = self.ce_loss(final_logits_b, y_b)
        
        return lam * loss_a + (1 - lam) * loss_b

# ==== DATASET (invariato) ====
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

# ==== TRAINING CON GeM + ArcFace + Center Loss + Mixup ====

def train_gem_arcface(model, train_loader):
    """
    Training avanzato con GeM pooling, ArcFace, Center Loss e Mixup.
    """
    print("[INFO] Starting training with GeM + ArcFace + Center Loss + Mixup...")
    
    num_classes = len(train_loader.dataset.label_map)
    
    # Loss functions
    arcface_loss_fn = ArcFaceLoss(scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN).to(device)
    center_loss_fn = CenterLoss(num_classes, EMBED_DIM).to(device)
    
    # Optimizer per parametri del modello
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.clip.visual.named_parameters() if p.requires_grad], 'lr': LR_BACKBONE},
        {'params': list(model.projection.parameters()) + list(model.classifier.parameters()) + list(model.gem_pooling.parameters()), 'lr': LR_BASE}
    ], weight_decay=1e-4)
    
    # Optimizer separato per i centri della center loss
    center_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=CENTER_LR)
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * (NUM_EPOCHS // 3), T_mult=1
    )
    
    scaler = GradScaler()
    
    model.train()
    center_loss_fn.train()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_arcface_loss = 0
        total_center_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Decide se applicare mixup
            use_mixup = random.random() < MIXUP_PROB
            
            optimizer.zero_grad()
            center_optimizer.zero_grad()
            
            with autocast():
                if use_mixup:
                    # Applica mixup
                    mixed_images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                    embeddings, _ = model(mixed_images)
                    
                    # ArcFace loss per mixup
                    arcface_loss = arcface_loss_fn.forward_mixup(
                        embeddings, y_a, y_b, lam, model.classifier.weight
                    )
                    
                    # Center loss (mix dei centri)
                    center_loss_a = center_loss_fn(embeddings, y_a)
                    center_loss_b = center_loss_fn(embeddings, y_b)
                    center_loss = lam * center_loss_a + (1 - lam) * center_loss_b
                    
                else:
                    # Training normale
                    embeddings, _ = model(images)
                    arcface_loss = arcface_loss_fn(embeddings, labels, model.classifier.weight)
                    center_loss = center_loss_fn(embeddings, labels)
                
                # Loss totale
                total_batch_loss = arcface_loss + CENTER_LOSS_WEIGHT * center_loss
            
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.step(center_optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += total_batch_loss.item()
            total_arcface_loss += arcface_loss.item()
            total_center_loss += center_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'ArcFace': f'{arcface_loss.item():.4f}',
                'Center': f'{center_loss.item():.4f}',
                'GeM_p': f'{model.gem_pooling.p.item():.3f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_arcface = total_arcface_loss / len(train_loader)
        avg_center = total_center_loss / len(train_loader)
        
        print(f'Epoch {epoch}: Total Loss = {avg_loss:.4f}, ArcFace = {avg_arcface:.4f}, Center = {avg_center:.4f}, GeM p = {model.gem_pooling.p.item():.3f}')
        
    print("[INFO] Training completed!")

# ==== ESTRAZIONE EMBEDDING E SUBMISSION (invariati) ====

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

    # Inizializza modello con nuova architettura
    num_classes = len(train_dataset.label_map)
    model = CLIPGeMArcFace(
        clip_model, embed_dim=EMBED_DIM, num_classes=num_classes, unfreeze_layers=UNFREEZE_LAYERS
    ).to(device)
    
    print(f"[INFO] Training with {num_classes} classes, embed_dim={EMBED_DIM}")
    print(f"[INFO] GeM pooling parameter p initialized to: {model.gem_pooling.p.item():.3f}")
    
    # Training con nuova architettura
    train_gem_arcface(model, train_loader)
    
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

if __name__ == "__main__":
    main()