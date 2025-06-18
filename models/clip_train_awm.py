import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_metric_learning.losses import ProxyAnchorLoss, TripletMarginLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.miners import TripletMarginMiner
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import open_clip
# from facenet_pytorch import MTCNN
from datetime import datetime

# ========== CONFIG ==========
BASE_DIR = "."
TRAIN_DIR = os.path.join(BASE_DIR, "train")
GALLERY_DIR = os.path.join(BASE_DIR, "test", "gallery")
QUERY_DIR = os.path.join(BASE_DIR, "test", "query")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EMBED_DIM = 2048
EPOCHS = 12
LR_BASE = 1e-4
LR_BACKBONE = 1e-5
MARGIN = 0.2
ALPHA = 64
CE_WEIGHT = 1.0
WARMUP_EPOCHS = 5
TOP_K = 5
SEED = 1

# Paths per salvataggio
MODEL_WEIGHTS_PATH = "andrea_model_weights.pth"
SUBMISSION_PATH = "submission_andrea.py"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)

print(f"[INFO] Using device: {DEVICE}")

# ========== FACE DETECTION (COMMENTED) ==========
# mtcnn = MTCNN(image_size=224, device=DEVICE)

# ========== MODEL ==========
class CLIPWithMLP(nn.Module):
    def __init__(self, base, out_dim, num_classes, unfreeze_layers=4):
        super().__init__()
        self.clip = base
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.unfreeze_layers = unfreeze_layers
        
        # Freeze/unfreeze layers
        total = len(base.visual.transformer.resblocks)
        for name, p in base.visual.named_parameters():
            if 'resblocks' in name:
                idx = int(name.split('.')[2])
                p.requires_grad = idx >= total - unfreeze_layers
            else:
                p.requires_grad = False
        
        self.head = nn.Sequential(
            nn.Linear(base.visual.output_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.15), nn.Linear(1024, out_dim)
        )
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        feat = self.clip.encode_image(x)
        emb = F.normalize(self.head(feat), dim=-1)
        logits = self.classifier(emb)
        return emb, logits
    
    def extract_features(self, x):
        """Metodo per estrarre solo gli embeddings (utile per ensemble)"""
        with torch.no_grad():
            feat = self.clip.encode_image(x)
            emb = F.normalize(self.head(feat), dim=-1)
        return emb
    
    def get_config(self):
        """Restituisce la configurazione del modello"""
        return {
            'out_dim': self.out_dim,
            'num_classes': self.num_classes,
            'unfreeze_layers': self.unfreeze_layers,
            'clip_model': 'ViT-L-14',
            'embed_dim': EMBED_DIM,
            'epochs': EPOCHS,
            'lr_base': LR_BASE,
            'lr_backbone': LR_BACKBONE,
            'batch_size': BATCH_SIZE,
            'timestamp': datetime.now().isoformat()
        }

# ========== DATASET FOR TEST (NO LABELS) ==========
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for img_file in os.listdir(root_dir):
            if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                self.samples.append(os.path.join(root_dir, img_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # # FACE DETECTION (COMMENTED)
        # try:
        #     face = mtcnn(image)
        #     if face is not None:
        #         face_np = face.permute(1, 2, 0).cpu().numpy()
        #         face_np = ((face_np + 1.0) / 2.0 * 255).astype(np.uint8)
        #         image = Image.fromarray(face_np)
        # except:
        #     pass  # Use original image if face detection fails
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path

# ========== LOADERS ==========
def make_loaders(prep):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.2,0.1),
        prep,
        transforms.RandomErasing(p=0.25, scale=(0.02,0.33), ratio=(0.3,3.3))
    ])
    
    # Training loader (with labels)
    tr_ds = ImageFolder(TRAIN_DIR, transform=train_tf)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                       sampler=MPerClassSampler(tr_ds.targets, m=4, length_before_new_iter=len(tr_ds)),
                       num_workers=4)
    
    # Test loaders (without labels)
    gal_ds = TestDataset(GALLERY_DIR, transform=prep)
    qry_ds = TestDataset(QUERY_DIR, transform=prep)
    gal_ld = DataLoader(gal_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    qry_ld = DataLoader(qry_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return tr_ld, qry_ld, gal_ld, tr_ds.classes

# ========== TRAINING ==========
def train_model(model, loader, num_classes):
    proxy = ProxyAnchorLoss(num_classes=num_classes, embedding_size=EMBED_DIM, margin=MARGIN, alpha=ALPHA).to(DEVICE)
    ce = nn.CrossEntropyLoss()
    miner = TripletMarginMiner(margin=0.2, type_of_triplets='hard')
    triplet_fn = TripletMarginLoss(margin=0.2)
    opt = torch.optim.AdamW([
        {'params':[p for n,p in model.clip.visual.named_parameters() if p.requires_grad], 'lr':LR_BACKBONE},
        {'params':list(model.head.parameters())+list(model.classifier.parameters()), 'lr':LR_BASE}
    ], weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=len(loader)*WARMUP_EPOCHS, T_mult=1)
    scaler = GradScaler()
    
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    for ep in range(1, EPOCHS+1):
        model.train(); total=0
        for imgs, labs in tqdm(loader, desc=f'Epoch {ep}/{EPOCHS}'):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            opt.zero_grad()
            with autocast():
                emb, logits = model(imgs)
                loss = proxy(emb, labs) + triplet_fn(emb, labs, miner(emb, labs)) + CE_WEIGHT*ce(logits, labs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        print(f'Epoch {ep}: Avg Loss = {total/len(loader):.4f}')
    
    # Fine-tune head/classifier
    print("[INFO] Fine-tuning head and classifier...")
    for p in model.clip.parameters(): p.requires_grad=False
    opt2 = torch.optim.AdamW(list(model.head.parameters())+list(model.classifier.parameters()), lr=LR_BASE*0.1, weight_decay=1e-4)
    scaler2 = GradScaler()
    for imgs, labs in tqdm(loader, desc='Fine-tune head'):
        imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
        opt2.zero_grad()
        with autocast():
            emb, logits = model(imgs)
            loss = proxy(emb, labs) + triplet_fn(emb, labs, miner(emb, labs)) + CE_WEIGHT*ce(logits, labs)
        scaler2.scale(loss).backward()
        scaler2.step(opt2); scaler2.update()
    
    print("[INFO] Training completed!")

# ========== EMBEDDING EXTRACTION ==========
def extract_embs(model, loader):
    model.eval(); embs = []; paths = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extract Embeddings'):
            imgs, img_paths = batch
            emb = model.extract_features(imgs.to(DEVICE))
            embs.append(emb.cpu())
            paths.extend(img_paths)
    return torch.vstack(embs), paths

# ========== SAVE MODEL ==========
def save_model_with_config(model, filepath):
    """Salva il modello con la sua configurazione"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'model_class': 'CLIPWithMLP'
    }
    torch.save(save_dict, filepath)
    print(f"✅ Model weights and config saved to: {filepath}")

# ========== SUBMISSION ==========
def generate_submission_py(qry_emb, gal_emb, qry_paths, gal_paths, K=5, out_path="submission.py"):
    """Genera submission in formato Python dictionary come Pretty Figures"""
    sims = qry_emb @ gal_emb.T
    ranks = np.argsort(-sims, axis=1)
    
    results = {}
    for i, qry_path in enumerate(qry_paths):
        top_k = ranks[i][:K]
        qry_filename = os.path.basename(qry_path).replace("\\", "/")
        matches = [os.path.basename(gal_paths[idx]).replace("\\", "/") for idx in top_k]
        results[qry_filename] = matches
    
    # Salva in formato Python dictionary
    with open(out_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            f.write(f'    "{key}": {json.dumps(value)},\n')
        f.write("}\n")
    
    print(f"✅ Submission saved to: {out_path}")

# ========== MAIN ==========
def main():
    print("[INFO] Loading CLIP model...")
    clip_model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model.to(DEVICE)

    print("[INFO] Creating data loaders...")
    tr_ld, qry_ld, gal_ld, classes = make_loaders(prep)
    print(f"[INFO] Found {len(classes)} classes in training data")

    print("[INFO] Initializing model...")
    model = CLIPWithMLP(clip_model, out_dim=EMBED_DIM, num_classes=len(classes)).to(DEVICE)

    print("[INFO] Starting training...")
    train_model(model, tr_ld, len(classes))

    print("[INFO] Saving model weights...")
    save_model_with_config(model, MODEL_WEIGHTS_PATH)

    print("[INFO] Extracting embeddings...")
    gal_emb, gal_paths = extract_embs(model, gal_ld)
    qry_emb, qry_paths = extract_embs(model, qry_ld)

    print("[INFO] Generating submission...")
    generate_submission_py(qry_emb.numpy(), gal_emb.numpy(), qry_paths, gal_paths, K=TOP_K, out_path=SUBMISSION_PATH)

    print("[INFO] Pipeline completed successfully!")

if __name__ == "__main__":
    main()