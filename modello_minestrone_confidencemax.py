import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== MODELLI ======
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

efficientnet_model = timm.create_model('tf_efficientnetv2_l', pretrained=True).to(device)
efficientnet_model.eval()

resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet_model.eval()

transform_eff = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
transform_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====== ESTRAZIONE EMBEDDING ======
def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def extract_clip_embeddings(image_paths):
    clip_model.eval()
    embeddings = []
    for path in tqdm(image_paths, desc="CLIP"):
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        embeddings.append(emb.cpu().numpy()[0])
    return l2_normalize(np.array(embeddings).astype("float32"))

def extract_dino_embeddings(image_paths):
    dino_model.eval()
    embeddings = []
    for path in tqdm(image_paths, desc="DINOv2"):
        image = Image.open(path).convert("RGB")
        inputs = dino_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = dino_model(**inputs)
            emb = output.last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu().numpy()[0])
    return l2_normalize(np.array(embeddings).astype("float32"))

def extract_efficientnet_embeddings(image_paths):
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="EfficientNetV2"):
        batch_paths = image_paths[i:i+32]
        imgs = [transform_eff(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = efficientnet_model.forward_features(batch)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1)
            feats = feats.squeeze(-1).squeeze(-1)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

def extract_resnet_embeddings(image_paths):
    feature_extractor = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="ResNet50"):
        batch_paths = image_paths[i:i+32]
        imgs = [transform_resnet(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

# ====== RETRIEVAL ======
def compute_similarity_scores(query_embs, gallery_embs):
    """Computes cosine similarity scores between query and gallery embeddings."""
    # Normalizzazione L2 per garantire che le similarità coseno siano tra -1 e 1
    query_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    gallery_norm = gallery_embs / np.linalg.norm(gallery_embs, axis=1, keepdims=True)
    
    # Calcola similarità coseno: sim(a,b) = dot(a,b)/(|a|*|b|)
    # Siccome abbiamo già normalizzato, possiamo semplicemente fare il prodotto scalare
    similarity = np.dot(query_norm, gallery_norm.T)
    
    # Converte in scores tra 0 e 1 (da -1,1 a 0,1)
    scores = (similarity + 1) / 2
    
    return scores

def retrieve_with_ensemble(query_files, gallery_files, model_scores_list, model_names, k=50, high_confidence_threshold=0.975):
    """
    Retrieves gallery images for each query using an ensemble of models.
    
    Args:
        query_files: Lista di file di query
        gallery_files: Lista di file della gallery
        model_scores_list: Lista di matrici di similarità [num_queries x num_gallery] per ogni modello
        model_names: Lista di nomi dei modelli per il logging
        k: Numero di immagini da recuperare per ogni query
        high_confidence_threshold: Soglia di confidenza per considerare un modello come "molto sicuro"
        
    Returns:
        Lista di risultati per ogni query
    """
    num_queries = len(query_files)
    num_models = len(model_scores_list)
    results = []
    
    for q_idx in range(num_queries):
        query_path = query_files[q_idx]
        query_rel = query_path.replace("\\", "/")
        
        # Estrai gli score di similarità per la query corrente da ogni modello
        current_scores = [model_scores[q_idx] for model_scores in model_scores_list]
        
        # Identifica se c'è un modello con confidenza superiore al threshold
        high_confidence_model = None
        high_confidence_score = 0
        high_confidence_indices = None
        
        for m_idx, scores in enumerate(current_scores):
            max_score = np.max(scores)
            if max_score > high_confidence_threshold:
                # Se troviamo più modelli con alta confidenza, prendiamo quello con lo score più alto
                if max_score > high_confidence_score:
                    high_confidence_score = max_score
                    high_confidence_model = m_idx
                    # Get indices in order of decreasing similarity
                    high_confidence_indices = np.argsort(scores)[::-1][:k]
        
        # Decidi se usare un singolo modello o la media
        if high_confidence_model is not None:
            print(f"Query {q_idx+1}/{num_queries}: Alta confidenza ({high_confidence_score:.4f}) dal modello {model_names[high_confidence_model]}")
            top_indices = high_confidence_indices
        else:
            # Media degli score di similarità
            avg_scores = np.mean(current_scores, axis=0)
            # Get indices in order of decreasing similarity
            top_indices = np.argsort(avg_scores)[::-1][:k]
            print(f"Query {q_idx+1}/{num_queries}: Usando media dei modelli")
        
        # Recupera le immagini della gallery
        gallery_matches = [gallery_files[idx].replace("\\", "/") for idx in top_indices]
        
        results.append({
            "filename": query_rel,
            "gallery_images": gallery_matches
        })
    
    return results

def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # ====== MAIN ======
if __name__ == "__main__":
    query_folder = "testing_images7_fish/test/query"
    gallery_folder = "testing_images7_fish/test/gallery"

    query_files = sorted([os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    gallery_files = sorted([os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    print("🔍 Estrazione degli embedding...")

    query_clip = extract_clip_embeddings(query_files)
    gallery_clip = extract_clip_embeddings(gallery_files)

    query_dino = extract_dino_embeddings(query_files)
    gallery_dino = extract_dino_embeddings(gallery_files)

    query_eff = extract_efficientnet_embeddings(query_files)
    gallery_eff = extract_efficientnet_embeddings(gallery_files)

    query_resnet = extract_resnet_embeddings(query_files)
    gallery_resnet = extract_resnet_embeddings(gallery_files)

    print(f"Shape query embeddings:")
    print(f" - CLIP        : {query_clip.shape}")
    print(f" - DINOv2      : {query_dino.shape}")
    print(f" - EfficientNet: {query_eff.shape}")
    print(f" - ResNet50    : {query_resnet.shape}")

    print("📊 Calcolo degli score di similarità per ogni modello...")
    
    # Calcola matrici di similarità per ogni modello
    clip_scores = compute_similarity_scores(query_clip, gallery_clip)
    dino_scores = compute_similarity_scores(query_dino, gallery_dino)
    eff_scores = compute_similarity_scores(query_eff, gallery_eff)
    resnet_scores = compute_similarity_scores(query_resnet, gallery_resnet)
    
    # Lista di tutte le matrici di score e nomi modelli
    all_model_scores = [clip_scores, dino_scores, eff_scores, resnet_scores]
    model_names = ["CLIP", "DINOv2", "EfficientNetV2", "ResNet50"]
    
    print("📥 Retrieval in corso con regola di confidenza...")
    # Alta confidenza se >0.95 (convertito da cosine similarity che va da 0 a 1)
    submission = retrieve_with_ensemble(query_files, gallery_files, all_model_scores, model_names, k=50, high_confidence_threshold=0.95)

    save_submission(submission, "submission/ensemble_confidence_based.json")
    print("✅ Submission salvata.")