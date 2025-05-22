import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights # Aggiunto per ConvNeXt
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel # AutoImageProcessor, AutoModel già presenti, usati anche per ViT
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# ====== DEFINIZIONE MODELLI E PROCESSOR/TRANSFORMER ======

# --- CLIP ---
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = None
clip_processor = None

# --- DINOv2 ---
dino_model_name = "facebook/dinov2-base"
dino_model = None
dino_processor = None

# --- EfficientNetV2 ---
efficientnet_model_name = 'tf_efficientnetv2_l'
efficientnet_model = None
transform_eff = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizzazione specifica per tf_efficientnet
])

# --- ResNet50 ---
resnet_model = None
transform_resnet = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- ConvNeXt ---
convnext_weights = None
convnext_model = None
preprocess_convnext = None # Verrà definito in init_convnext_model

# --- ViT (Vision Transformer by Google) ---
vit_model_name = "google/vit-base-patch16-224"
vit_model = None
vit_processor = None

# ====== FUNZIONI DI INIZIALIZZAZIONE MODELLI (Lazy Loading) ======
def init_clip_model():
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Caricamento CLIP model: {clip_model_name}")
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        clip_model.eval()

def init_dino_model():
    global dino_model, dino_processor
    if dino_model is None:
        print(f"Caricamento DINOv2 model: {dino_model_name}")
        dino_model = AutoModel.from_pretrained(dino_model_name).to(device)
        dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        dino_model.eval()

def init_efficientnet_model():
    global efficientnet_model
    if efficientnet_model is None:
        print(f"Caricamento EfficientNetV2 model: {efficientnet_model_name}")
        efficientnet_model = timm.create_model(efficientnet_model_name, pretrained=True).to(device)
        efficientnet_model.eval()

def init_resnet_model():
    global resnet_model
    if resnet_model is None:
        print("Caricamento ResNet50 model")
        resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
        resnet_model.eval()

def init_convnext_model():
    global convnext_model, convnext_weights, preprocess_convnext
    if convnext_model is None:
        print("Caricamento ConvNeXt model")
        convnext_weights = ConvNeXt_Base_Weights.DEFAULT
        convnext_model = convnext_base(weights=convnext_weights).to(device)
        convnext_model.eval()
        # Preprocessing specifico per ConvNeXt dai pesi, di solito include ToTensor e Normalizzazione
        # Assicuriamoci che Resize e CenterCrop siano coerenti con gli altri modelli se possibile
        preprocess_convnext = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), # Standardizzato
            transforms.CenterCrop(224), # Standardizzato
            convnext_weights.transforms() # Applica le trasformazioni dei pesi (ToTensor, Normalize, etc.)
        ])


def init_vit_model():
    global vit_model, vit_processor
    if vit_model is None:
        print(f"Caricamento ViT model: {vit_model_name}")
        vit_model = AutoModel.from_pretrained(vit_model_name).to(device)
        vit_processor = AutoImageProcessor.from_pretrained(vit_model_name)
        vit_model.eval()

# Dizionario per mappare i nomi dei modelli alle loro funzioni di inizializzazione
MODEL_INITIALIZERS = {
    "CLIP": init_clip_model,
    "DINOv2": init_dino_model,
    "EfficientNetV2": init_efficientnet_model,
    "ResNet50": init_resnet_model,
    "ConvNeXt": init_convnext_model,
    "ViT": init_vit_model,
}

# ====== ESTRAZIONE EMBEDDING ======
def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def extract_clip_embeddings(image_paths):
    # init_clip_model() # Assicura che il modello sia caricato
    embeddings = []
    for path in tqdm(image_paths, desc="CLIP"):
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        embeddings.append(emb.cpu().numpy()[0])
    return l2_normalize(np.array(embeddings).astype("float32"))

def extract_dino_embeddings(image_paths):
    # init_dino_model()
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
    # init_efficientnet_model()
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
    # init_resnet_model()
    # Il feature_extractor deve essere creato dopo che resnet_model è inizializzato
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

def extract_convnext_embeddings(image_paths):
    # init_convnext_model()
    # Il feature_extractor deve essere creato dopo che convnext_model è inizializzato
    # Estrae features prima del layer di classificazione finale
    feature_extractor_convnext = torch.nn.Sequential(convnext_model.features, convnext_model.avgpool)
    embeddings = []
    for i in tqdm(range(0, len(image_paths), 32), desc="ConvNeXt"): # Aggiunto batching
        batch_paths = image_paths[i:i+32]
        imgs = [preprocess_convnext(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = feature_extractor_convnext(batch)
            feats = feats.view(feats.size(0), -1) # Flatten (B, C, 1, 1) to (B, C)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))


def extract_vit_embeddings(image_paths):
    # init_vit_model()
    embeddings = []
    # ViT processor può gestire batch di immagini, ma qui manteniamo il ciclo per coerenza di loading
    # oppure si può passare una lista di PIL Image direttamente al processor se si preferisce.
    # Per semplicità e per seguire il pattern degli altri, facciamo batching manuale.
    for i in tqdm(range(0, len(image_paths), 32), desc="ViT"): # Aggiunto batching
        batch_paths = image_paths[i:i+32]
        # Il processor di Hugging Face di solito gestisce una lista di immagini PIL
        batch_pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = vit_processor(images=batch_pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            # Estrai il CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu()) # emb è già un tensore, non un array numpy qui
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

# Dizionario per mappare i nomi dei modelli alle loro funzioni di estrazione embedding
MODEL_EXTRACTORS = {
    "CLIP": extract_clip_embeddings,
    "DINOv2": extract_dino_embeddings,
    "EfficientNetV2": extract_efficientnet_embeddings,
    "ResNet50": extract_resnet_embeddings,
    "ConvNeXt": extract_convnext_embeddings,
    "ViT": extract_vit_embeddings,
}

# ====== RETRIEVAL ======
def compute_similarity_scores(query_embs, gallery_embs):
    # Gli embedding dovrebbero essere già normalizzati L2 dalle funzioni di estrazione
    similarity = np.dot(query_embs, gallery_embs.T)
    scores = (similarity + 1) / 2 # Mappa da [-1, 1] a [0, 1]
    return scores

def retrieve_with_weighted_ensemble(query_files, gallery_files, model_scores_list, model_accuracies, model_names, k=50):
    num_queries = len(query_files)
    results = []
    
    accuracies_array = np.array(model_accuracies)
    if sum(accuracies_array) == 0:
        weights = np.ones(len(model_accuracies)) / len(model_accuracies)
        print("Attenzione: tutte le accuratezze fornite sono 0. Si utilizzano pesi uguali.")
    else:
        weights = accuracies_array / sum(accuracies_array)

    print(f"Pesi dei modelli per la media ponderata (basati su {len(model_names)} modelli selezionati):")
    for name, weight, accuracy in zip(model_names, weights, model_accuracies):
        print(f" - {name}: {weight:.4f} (accuratezza fornita: {accuracy:.2f}%)")
    
    for q_idx in range(num_queries):
        query_path = query_files[q_idx]
        query_rel = query_path.replace("\\", "/")
        
        current_scores_for_query = [model_scores[q_idx] for model_scores in model_scores_list]
        
        weighted_scores = np.zeros_like(current_scores_for_query[0], dtype=np.float32)
        for i, scores_single_model_for_query in enumerate(current_scores_for_query):
            weighted_scores += scores_single_model_for_query * weights[i]
            
        top_indices = np.argsort(weighted_scores)[::-1][:k]
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
    query_folder = "testing_images6_clothes/test/query" # Modifica se necessario
    gallery_folder = "testing_images6_clothes/test/gallery" # Modifica se necessario

    # ====== CONFIGURAZIONE MODELLI DA UTILIZZARE ======
    # Specifica qui quali modelli vuoi includere. I nomi devono corrispondere alle chiavi in MODEL_INITIALIZERS e MODEL_EXTRACTORS.
    # Esempi:
    # ENABLED_MODELS = ["CLIP", "DINOv2"]
    # ENABLED_MODELS = ["ResNet50", "EfficientNetV2", "ConvNeXt", "ViT"]
    # ENABLED_MODELS = ["CLIP", "DINOv2", "EfficientNetV2", "ResNet50", "ConvNeXt", "ViT"] # Tutti i modelli
    ENABLED_MODELS = ["CLIP", "ViT", "ConvNeXt"] # Esempio: solo alcuni modelli

    # Accuratezze di default (come percentuali, es. 94.0 per 94%)
    # Queste verranno usate per i modelli in ENABLED_MODELS.
    # Puoi aggiornarle qui se hai stime più recenti.
    MODEL_DEFAULT_ACCURACIES = {
        "CLIP": 56.71,
        "DINOv2": 43.16,
        "EfficientNetV2": 54.42,
        "ResNet50": 50.00,
        "ConvNeXt": 52.50, # Esempio, da aggiornare con valore reale
        "ViT": 55.20,     # Esempio, da aggiornare con valore reale
    }
    # =====================================================

    # Validazione dei modelli scelti
    for model_name in ENABLED_MODELS:
        if model_name not in MODEL_INITIALIZERS or model_name not in MODEL_EXTRACTORS:
            raise ValueError(f"Modello '{model_name}' non configurato correttamente. Controlla MODEL_INITIALIZERS e MODEL_EXTRACTORS.")
        if model_name not in MODEL_DEFAULT_ACCURACIES:
            print(f"Attenzione: Accuratezza di default non specificata per '{model_name}'. Verrà usato 0.0, considera di aggiungerla a MODEL_DEFAULT_ACCURACIES.")
            MODEL_DEFAULT_ACCURACIES[model_name] = 0.0 # O gestisci diversamente

    if not ENABLED_MODELS:
        print("Nessun modello selezionato in ENABLED_MODELS. Uscita.")
        exit()
    
    print(f"Modelli selezionati per l'ensemble: {', '.join(ENABLED_MODELS)}")

    query_files = sorted([os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    gallery_files = sorted([os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not query_files or not gallery_files:
        print(f"Errore: cartella query ({query_folder}) o gallery ({gallery_folder}) vuota o senza immagini supportate.")
        exit()
    
    print(f"Trovate {len(query_files)} immagini query e {len(gallery_files)} immagini gallery.")

    # Inizializzazione dei modelli selezionati
    for model_name in ENABLED_MODELS:
        MODEL_INITIALIZERS[model_name]() # Carica modello e processor/transform

    active_query_embeddings = {}
    active_gallery_embeddings = {}
    
    print("\n🔍 Estrazione degli embedding per i modelli selezionati...")
    for model_name in ENABLED_MODELS:
        print(f"--- Estrazione per {model_name} ---")
        extractor_fn = MODEL_EXTRACTORS[model_name]
        active_query_embeddings[model_name] = extractor_fn(query_files)
        active_gallery_embeddings[model_name] = extractor_fn(gallery_files)

    print("\nShape query embeddings:")
    for model_name, embeddings in active_query_embeddings.items():
        print(f" - {model_name:<15}: {embeddings.shape}")

    active_model_scores = []
    active_model_names_for_ensemble = []
    active_model_accuracies_for_ensemble = []

    print("\n📊 Calcolo degli score di similarità per ogni modello selezionato...")
    for model_name in ENABLED_MODELS:
        scores = compute_similarity_scores(active_query_embeddings[model_name], active_gallery_embeddings[model_name])
        active_model_scores.append(scores)
        active_model_names_for_ensemble.append(model_name)
        active_model_accuracies_for_ensemble.append(MODEL_DEFAULT_ACCURACIES[model_name])
        print(f" - Score calcolati per {model_name}")

    if not active_model_scores:
        print("Nessuno score calcolato. Impossibile procedere con il retrieval.")
        exit()

    print("\n📥 Retrieval in corso con media ponderata basata su accuratezze fornite...")
    
    submission = retrieve_with_weighted_ensemble(
        query_files, 
        gallery_files, 
        active_model_scores, 
        active_model_accuracies_for_ensemble,
        active_model_names_for_ensemble, 
        k=50 # Numero di immagini da recuperare
    )

    output_filename = "submission/ensemble_manual_accuracies.json"
    save_submission(submission, output_filename)
    print(f"✅ Submission salvata in: {output_filename}")