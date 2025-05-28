import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import faiss
import requests
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === FUNZIONI PER FINE-TUNING ===
def prepare_model_for_finetuning(model, trainable_layer_names=["block8", "last_linear"]):
    for name, param in model.named_parameters():
        param.requires_grad = False
        for layer_name in trainable_layer_names:
            if layer_name in name:
                param.requires_grad = True
    print("Trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")
    return model

class FaceNetClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(FaceNetClassifier, self).__init__()
        self.base = base_model
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.set_grad_enabled(self.training):
            features = self.base(x)
        logits = self.classifier(features)
        return logits

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, mtcnn):
        self.samples = []
        self.mtcnn = mtcnn
        self.label_map = {}
        self.image_paths = []
        self.labels = []

        identities = sorted(os.listdir(root_folder))
        self.label_map = {name: idx for idx, name in enumerate(identities)}

        for identity in identities:
            identity_path = os.path.join(root_folder, identity)
            if not os.path.isdir(identity_path):
                continue
            for fname in os.listdir(identity_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(identity_path, fname))
                    self.labels.append(self.label_map[identity])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        img_tensor = detect_and_crop_face(path, self.mtcnn)
        if img_tensor is None:
            img_tensor = torch.zeros(3, 160, 160)

        return img_tensor, label

def train_finetune_model(model, classifier, dataloader, num_epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=lr)

    classifier.to(device)
    classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={correct/total:.4f}")

# === MTCNN PER FACE DETECTION ===
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True,
    device=device
)

# === FACENET MODEL LOADING ===
try:
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("FaceNet model loaded successfully")
except Exception as e:
    print(f"Errore nel caricamento del modello FaceNet: {e}")
    exit()

# === FINE-TUNING ===
print("Preparazione fine-tuning...")
model_frozen = prepare_model_for_finetuning(model)

train_folder = "train"
train_dataset = FaceDataset(train_folder, mtcnn)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

num_classes = len(set(train_dataset.labels))
classifier = FaceNetClassifier(model_frozen, num_classes)

print(f"Inizio fine-tuning su {num_classes} classi...")
train_finetune_model(model_frozen, classifier, train_loader, num_epochs=5)
print("✅ Fine-tuning completato.")

# === FUNZIONI ORIGINALI (come da file precedente) ===

def detect_and_crop_face(image_path, mtcnn_detector):
    """Detect and crop face from image using MTCNN"""
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Detect face
        img_cropped = mtcnn_detector(img)
        
        if img_cropped is not None:
            return img_cropped
        else:
            # Fallback: resize original image if no face detected
            img_tensor = transforms.functional.to_tensor(img)
            img_tensor = transforms.functional.resize(img_tensor, (160, 160))
            return img_tensor
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def augment_image(img_tensor):
    """Apply data augmentation to face image"""
    augmentations = []
    
    # Original
    augmentations.append(img_tensor)
    
    # Horizontal flip
    augmentations.append(torch.flip(img_tensor, [2]))
    
    # Slight rotation (-5 to 5 degrees)
    angle = np.random.uniform(-5, 5)
    augmentations.append(transforms.functional.rotate(img_tensor, angle))
    
    # Brightness adjustment
    brightness_factor = np.random.uniform(0.9, 1.1)
    augmentations.append(transforms.functional.adjust_brightness(img_tensor, brightness_factor))
    
    return augmentations

def get_feature_extractor_improved(base_model, use_mtcnn=True, use_augmentation=False):
    def extractor(image_paths, is_training=False):
        embs = []
        processed_paths = []
        
        for path in tqdm(image_paths, desc="Estrazione feature (FaceNet Enhanced)"):
            try:
                if use_mtcnn:
                    img_tensor = detect_and_crop_face(path, mtcnn)
                else:
                    img = Image.open(path).convert("RGB")
                    img_tensor = transforms.functional.to_tensor(img)
                    img_tensor = transforms.functional.resize(img_tensor, (160, 160))
                
                if img_tensor is None:
                    continue
                
                img_tensor = img_tensor.to(device)
                
                # Apply augmentation for training data
                if use_augmentation and is_training:
                    augmented_images = augment_image(img_tensor)
                    embeddings_for_image = []
                    
                    for aug_img in augmented_images:
                        with torch.no_grad():
                            emb = base_model(aug_img.unsqueeze(0)).cpu().numpy()[0]
                            embeddings_for_image.append(emb)
                    
                    # Average the embeddings from augmented images
                    avg_emb = np.mean(embeddings_for_image, axis=0)
                    embs.append(avg_emb)
                else:
                    with torch.no_grad():
                        emb = base_model(img_tensor.unsqueeze(0)).cpu().numpy()[0]
                        embs.append(emb)
                
                processed_paths.append(path)
                
            except Exception as e:
                print(f"Errore con {path}: {e}")
                continue
                
        return np.array(embs).astype("float32") if embs else np.array([]).astype("float32"), processed_paths
    
    return extractor

def build_training_embeddings(train_folder, feature_extractor_fn):
    """Build embeddings for training data organized by identity"""
    identity_embeddings = {}
    identity_paths = {}
    
    for identity_folder in tqdm(os.listdir(train_folder), desc="Processing training identities"):
        identity_path = os.path.join(train_folder, identity_folder)
        if not os.path.isdir(identity_path):
            continue
            
        image_files = [
            os.path.join(identity_path, f) 
            for f in os.listdir(identity_path) 
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        if image_files:
            embs, processed_paths = feature_extractor_fn(image_files, is_training=True)
            if len(embs) > 0:
                # Average embeddings for this identity
                avg_embedding = np.mean(embs, axis=0)
                identity_embeddings[identity_folder] = avg_embedding
                identity_paths[identity_folder] = processed_paths
    
    return identity_embeddings, identity_paths

def enhanced_retrieval_with_training(query_embs, query_files, gallery_embs, gallery_files, 
                                   identity_embeddings, k=10, use_reranking=True):
    """Enhanced retrieval using training data knowledge"""
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Embedding vuoti. Retrieval non eseguibile.")
        return []

    # Normalizzazione L2
    faiss.normalize_L2(gallery_embs)
    faiss.normalize_L2(query_embs)
    
    # Build identity embeddings matrix
    if identity_embeddings:
        identity_names = list(identity_embeddings.keys())
        identity_embs = np.array([identity_embeddings[name] for name in identity_names]).astype("float32")
        faiss.normalize_L2(identity_embs)
    
    # Standard gallery search
    dim = gallery_embs.shape[1]
    gallery_index = faiss.IndexFlatIP(dim)
    gallery_index.add(gallery_embs)
    
    # Identity search if available
    if identity_embeddings:
        identity_index = faiss.IndexFlatIP(dim)
        identity_index.add(identity_embs)
    
    results = []
    
    for i, query_path in enumerate(query_files):
        query_emb = query_embs[i:i+1]
        
        # Search in gallery
        gallery_distances, gallery_indices = gallery_index.search(query_emb, min(k*2, gallery_embs.shape[0]))
        
        # Search in identity embeddings for additional context
        if identity_embeddings and use_reranking:
            identity_distances, identity_indices = identity_index.search(query_emb, min(5, len(identity_names)))
            
            # Get top matching identities
            top_identities = [identity_names[idx] for idx in identity_indices[0]]
            
            # Re-rank gallery results based on identity similarity
            gallery_matches = []
            for idx in gallery_indices[0]:
                gallery_path = gallery_files[idx]
                gallery_filename = os.path.basename(gallery_path)
                
                # Check if gallery image belongs to a top-matching identity
                identity_bonus = 0
                for j, identity in enumerate(top_identities):
                    if identity.lower() in gallery_filename.lower():
                        identity_bonus = 0.1 * (1 - j/len(top_identities))  # Bonus decreases with rank
                        break
                
                original_score = gallery_distances[0][len(gallery_matches)]
                adjusted_score = original_score + identity_bonus
                
                gallery_matches.append((gallery_path, adjusted_score))
            
            # Sort by adjusted score
            gallery_matches.sort(key=lambda x: x[1], reverse=True)
            final_matches = [match[0] for match in gallery_matches[:k]]
        else:
            final_matches = [gallery_files[idx] for idx in gallery_indices[0][:k]]
        
        query_rel = query_path.replace("\\", "/")
        final_matches_rel = [match.replace("\\", "/") for match in final_matches]
        
        results.append({
            "filename": query_rel,
            "gallery_images": final_matches_rel
        })
    
    return results

def save_submission_d(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for key, value in results.items():
            f.write(f'    "{key}": {json.dumps(value)},\n')
        f.write("}\n")

def submit(results, groupname, url=" http://tatooine.disi.unitn.it:3001/retrieval/"):
    res = {}
    res["groupname"] = groupname
    res["images"] = results
    res = json.dumps(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
        return result['accuracy']
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        return None

# === CONFIGURAZIONE PATHS ===
train_folder = "train"
query_folder = "test/query"
gallery_folder = "test/gallery"

# === CARICAMENTO FILE ===
try:
    query_files = [os.path.join(query_folder, f) for f in os.listdir(query_folder) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    gallery_files = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
except Exception as e:
    print(f"Errore nel caricamento immagini: {e}")
    exit()

if not query_files or not gallery_files:
    print("Query o gallery vuote.")
    exit()

print(f"Trovate {len(query_files)} query e {len(gallery_files)} gallery.")

# === ESTRAZIONE FEATURE CON MIGLIORAMENTI ===
feature_extractor_fn = get_feature_extractor_improved(model_frozen, use_mtcnn=True, use_augmentation=True)

# Build training embeddings
print("Building training embeddings...")
identity_embeddings, identity_paths = build_training_embeddings(train_folder, feature_extractor_fn)
print(f"Created embeddings for {len(identity_embeddings)} identities")

# Extract query and gallery embeddings
print("Extracting query embeddings...")
query_embs, processed_query_files = feature_extractor_fn(query_files, is_training=False)

print("Extracting gallery embeddings...")
gallery_embs, processed_gallery_files = feature_extractor_fn(gallery_files, is_training=False)

# === RETRIEVAL E SALVATAGGIO ===
if query_embs.shape[0] > 0 and gallery_embs.shape[0] > 0:
    print("Performing enhanced retrieval...")
    submission_list = enhanced_retrieval_with_training(
        query_embs, processed_query_files, 
        gallery_embs, processed_gallery_files, 
        identity_embeddings, k=10, use_reranking=True
    )
    
    data = {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }
    
    submission_path = "submission/submission_facenet_enhanced.py"
    save_submission_d(data, submission_path)
    
    submit(data, "Pretty Figure")
    accuracy = submit(data, "Pretty Figure")
    
    print(f"✅ Submission salvata in: {submission_path}")
    if accuracy:
        print(f"🎯 Accuracy ottenuta: {accuracy}")
else:
    print("Embedding non estratti. Retrieval saltato.")