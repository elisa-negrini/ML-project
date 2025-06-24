import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
# PCA è stato rimosso # from sklearn.decomposition import PCA
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

# ====== MEDIA DEGLI EMBEDDING CON PCA ======
# La funzione mean_fuse_embeddings è stata rimossa come richiesto.

# ====== RETRIEVAL ======
def compute_similarity_scores(query_embs, gallery_embs):
    """Computes cosine similarity scores between query and gallery embeddings."""
    # Normalizzazione L2 per garantire che le similarità coseno siano tra -1 e 1
    # Se gli embedding sono già normalizzati L2, questa è ridondante ma innocua.
    # Le funzioni di estrazione embedding includono già l2_normalize.
    query_norm = query_embs # query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    gallery_norm = gallery_embs # gallery_embs / np.linalg.norm(gallery_embs, axis=1, keepdims=True)
    
    # Calcola similarità coseno: sim(a,b) = dot(a,b)/(|a|*|b|)
    # Siccome abbiamo già normalizzato, possiamo semplicemente fare il prodotto scalare
    similarity = np.dot(query_norm, gallery_norm.T)
    
    # Converte in scores tra 0 e 1 (da -1,1 a 0,1 assumendo che la similarità coseno sia già in [-1, 1])
    # Se gli embedding sono normalizzati L2, il prodotto scalare è la similarità coseno.
    scores = (similarity + 1) / 2 
    
    return scores

def retrieve_with_weighted_ensemble(query_files, gallery_files, model_scores_list, model_accuracies, model_names, k=50):
    """
    Retrieves gallery images for each query using a weighted ensemble of models.
    
    Args:
        query_files: Lista di file di query
        gallery_files: Lista di file della gallery
        model_scores_list: Lista di matrici di similarità [num_queries x num_gallery] per ogni modello
        model_accuracies: Lista di accuratezze (es. percentuali come 94.0, 80.0) per ogni modello (per la ponderazione)
        model_names: Lista di nomi dei modelli per il logging
        k: Numero di immagini da recuperare per ogni query
        
    Returns:
        Lista di risultati per ogni query
    """
    num_queries = len(query_files)
    # num_models = len(model_scores_list) # Non usato
    results = []
    
    # Normalizza i pesi in modo che la somma sia 1
    # Assicura che model_accuracies sia un array numpy per operazioni vettoriali
    accuracies_array = np.array(model_accuracies)
    if sum(accuracies_array) == 0: # Evita divisione per zero se tutte le accuratezze sono 0
        weights = np.ones(len(model_accuracies)) / len(model_accuracies) # Pesi uguali
        print("Attenzione: tutte le accuratezze fornite sono 0. Si utilizzano pesi uguali.")
    else:
        weights = accuracies_array / sum(accuracies_array)

    print(f"Pesi dei modelli per la media ponderata:")
    for name, weight, accuracy in zip(model_names, weights, model_accuracies):
        print(f" - {name}: {weight:.4f} (accuratezza fornita: {accuracy:.2f}%)")
    
    for q_idx in range(num_queries):
        query_path = query_files[q_idx]
        query_rel = query_path.replace("\\", "/")
        
        # Estrai gli score di similarità per la query corrente da ogni modello
        current_scores = [model_scores[q_idx] for model_scores in model_scores_list]
        
        # Media ponderata degli score di similarità
        weighted_scores = np.zeros_like(current_scores[0], dtype=np.float32) # Specifica dtype per coerenza
        for i, scores in enumerate(current_scores):
            weighted_scores += scores * weights[i]
        
        # Get indices in order of decreasing similarity
        top_indices = np.argsort(weighted_scores)[::-1][:k]
        
        # Mostra la distribuzione dei pesi per questa query (opzionale)
        if q_idx < 5:  # Mostra solo per le prime 5 query per non inondare il terminale
            # I pesi sono gli stessi per tutte le query, quindi potrebbe essere ridondante stamparlo qui per query
            # print(f"Query {q_idx+1}/{num_queries}: Pesi modelli {weights}") 
            pass # La stampa dei pesi è già stata fatta sopra, una volta sola.
        elif q_idx == 5:
            # print("...") # Non necessario se non stampiamo i pesi per query
            pass
        
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
    query_folder = "testing_images6_clothes/test/query"
    gallery_folder = "testing_images6_clothes/test/gallery"

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
    
    # ==============================================================================
    # MODIFICA QUI LE ACCURATEZZE DEI MODELLI COME PERCENTUALI (es. 94.0 per 94%)
    # L'ordine deve corrispondere a quello in model_names e all_model_scores
    # ==============================================================================
    model_accuracies = [56.71, 43.16, 54.42, 50] # Esempio: CLIP 92.5%, DINOv2 91.0%, etc.
    # Assicurati che il numero di elementi in model_accuracies corrisponda al numero di modelli
    if len(model_accuracies) != len(model_names):
        raise ValueError("Il numero di accuratezze fornite non corrisponde al numero di modelli.")

    print("📥 Retrieval in corso con media ponderata basata su accuratezze fornite...")
    
    submission = retrieve_with_weighted_ensemble(
        query_files, 
        gallery_files, 
        all_model_scores, 
        model_accuracies, # Passa le accuratezze definite manualmente
        model_names, 
        k=50
    )

    save_submission(submission, "submission/ensemble_manual_accuracies.json")
    print("✅ Submission salvata.")