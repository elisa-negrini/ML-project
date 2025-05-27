import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, ResNet50_Weights
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
import timm
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# Dimensione del batch per l'estrazione degli embedding
BATCH_SIZE = 32 # Puoi modificare questo valore se necessario

# ====== CONFIGURAZIONE FINE-TUNING ======
FINETUNING_CONFIG = {
    "ENABLED": True,  # Master switch for fine-tuning
    "TRAIN_DATA_DIR": "testing_images5/training", # MODIFICA: Percorso alla tua cartella di training (es. 'path/to/train_data')
                                                 # Deve avere sottocartelle per ogni classe (es. train_data/class1, train_data/class2)
    "FINETUNE_BATCH_SIZE": 8,
    "NUM_EPOCHS": 3,    # Numero di epoche per il fine-tuning
    "LEARNING_RATE": 1e-4, # Learning rate per il classificatore
    "BASE_MODEL_LEARNING_RATE": 1e-5, # Learning rate più basso per i layer del base_model
    "MODELS_TO_FINETUNE": { # Specifica quali modelli abilitati verranno fine-tuned
        "CLIP": True,
        "DINOv2": True,
        "EfficientNetV2": True,
        "ResNet50": True,
        "ConvNeXt": True,
        "ViT": True,
    }
}
NUM_CLASSES = None # Verrà derivato dal dataset di training

# ====== DEFINIZIONE MODELLI E PROCESSOR/TRANSFORMER ======

# --- CLIP ---
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = None
clip_processor = None
clip_embed_dim = 512

# --- DINOv2 ---
dino_model_name = "facebook/dinov2-base"
dino_model = None
dino_processor = None
dino_embed_dim = 768 # For dinov2-base

# --- EfficientNetV2 ---
efficientnet_model_name = 'tf_efficientnetv2_l'
efficientnet_model = None
efficientnet_embed_dim = 1280 # For tf_efficientnetv2_l
transform_eff = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC), # Adjusted for effnetv2_l common practice
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Timm models often use 0.5, 0.5
])

# --- ResNet50 ---
resnet_model = None
resnet_embed_dim = 2048 # For ResNet50 (output of avgpool)
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
preprocess_convnext = None
convnext_embed_dim = 1024 # For ConvNeXt Base (output of avgpool)

# --- ViT (Vision Transformer by Google) ---
vit_model_name = "google/vit-base-patch16-224"
vit_model = None
vit_processor = None
vit_embed_dim = 768 # For vit-base

MODEL_EMBED_DIMS = {
    "CLIP": clip_embed_dim,
    "DINOv2": dino_embed_dim,
    "EfficientNetV2": efficientnet_embed_dim,
    "ResNet50": resnet_embed_dim,
    "ConvNeXt": convnext_embed_dim,
    "ViT": vit_embed_dim,
}
MODEL_TRANSFORMS_FOR_TRAINING = {
    "CLIP": None, # Gestito da clip_processor nel training loop
    "DINOv2": None, # Gestito da dino_processor nel training loop
    "EfficientNetV2": transform_eff,
    "ResNet50": transform_resnet,
    "ConvNeXt": None, # Verrà definito in init_convnext_model e usato
    "ViT": None, # Gestito da vit_processor nel training loop
}


# ====== FUNZIONI DI INIZIALIZZAZIONE MODELLI (Lazy Loading) ======
def init_clip_model():
    global clip_model, clip_processor
    if clip_model is None:
        print(f"Caricamento CLIP model: {clip_model_name}")
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # CORREZIONE QUI: Usa crop_size per ottenere le dimensioni corrette
        image_size = clip_processor.image_processor.crop_size["height"] # Solitamente è un valore quadrato (es. 224)
        
        MODEL_TRANSFORMS_FOR_TRAINING["CLIP"] = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        clip_model.eval()

def init_dino_model():
    global dino_model, dino_processor
    if dino_model is None:
        print(f"Caricamento DINOv2 model: {dino_model_name}")
        dino_model = AutoModel.from_pretrained(dino_model_name).to(device)
        dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)

        # CORREZIONE QUI: Usa crop_size
        try:
            image_size = dino_processor.crop_size["height"]
        except AttributeError:
            print(f"Attenzione: dino_processor.crop_size non trovato per {dino_model_name}. Tento con dino_processor.size (potrebbe essere un int).")
            print(f"DEBUG: dino_processor.size = {dino_processor.size}, type: {type(dino_processor.size)}") # AGGIUNGI QUESTO
            if isinstance(dino_processor.size, int):
                image_size = dino_processor.size
            elif isinstance(dino_processor.size, dict) and 'shortest_edge' in dino_processor.size:
                image_size = dino_processor.size['shortest_edge']
            else:
                image_size = 224 # Valore di fallback, DA VERIFICARE SE NECESSARIO
            print(f"Usando fallback image_size={image_size} per DINOv2 transforms.")

        MODEL_TRANSFORMS_FOR_TRAINING["DINOv2"] = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        dino_model.eval()

def init_efficientnet_model():
    global efficientnet_model
    if efficientnet_model is None:
        print(f"Caricamento EfficientNetV2 model: {efficientnet_model_name}")
        efficientnet_model = timm.create_model(efficientnet_model_name, pretrained=True, num_classes=0).to(device) # num_classes=0 for feature extraction
        efficientnet_model.eval()


def init_resnet_model():
    global resnet_model
    if resnet_model is None:
        print("Caricamento ResNet50 model")
        resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        resnet_model.eval()

def init_convnext_model():
    global convnext_model, convnext_weights, preprocess_convnext
    if convnext_model is None:
        print("Caricamento ConvNeXt model")
        convnext_weights = ConvNeXt_Base_Weights.DEFAULT
        convnext_model = convnext_base(weights=convnext_weights).to(device)
        convnext_model.eval()
        # Transform per embedding (usato anche per training se non sovrascritto)
        preprocess_convnext = convnext_weights.transforms()
        MODEL_TRANSFORMS_FOR_TRAINING["ConvNeXt"] = preprocess_convnext
        # Non è necessario controllare se ToTensor è presente perché preprocess_convnext è già la composizione corretta
        # For default ConvNeXt_Base_Weights.DEFAULT.transforms(), it's usually a complete transform pipeline.
        # Rimuovi la logica di controllo e ricomposizione:
        # if not any(isinstance(t, transforms.ToTensor) for t in preprocess_convnext.transforms):
        #     MODEL_TRANSFORMS_FOR_TRAINING["ConvNeXt"] = transforms.Compose(
        #         [t for t in preprocess_convnext.transforms if not isinstance(t, transforms.Normalize)] +
        #         [transforms.ToTensor(),
        #          next(t for t in preprocess_convnext.transforms if isinstance(t, transforms.Normalize))]
        #     )

def init_vit_model():
    global vit_model, vit_processor
    if vit_model is None:
        print(f"Caricamento ViT model: {vit_model_name}")
        vit_model = AutoModel.from_pretrained(vit_model_name).to(device)
        vit_processor = AutoImageProcessor.from_pretrained(vit_model_name)

        # MODIFICA PROATTIVA QUI: Usa crop_size
        try:
            image_size = vit_processor.crop_size["height"]
        except AttributeError:
            print(f"Attenzione: vit_processor.crop_size non trovato per {vit_model_name}. Tento con vit_processor.size.")
            image_size = 224 # Valore di fallback, DA VERIFICARE SE NECESSARIO
            print(f"Usando fallback image_size={image_size} per ViT transforms.")


        MODEL_TRANSFORMS_FOR_TRAINING["ViT"] = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        vit_model.eval()

MODEL_INITIALIZERS = {
    "CLIP": init_clip_model,
    "DINOv2": init_dino_model,
    "EfficientNetV2": init_efficientnet_model,
    "ResNet50": init_resnet_model,
    "ConvNeXt": init_convnext_model,
    "ViT": init_vit_model,
}

MODEL_PROCESSORS = {
    "CLIP": lambda: clip_processor,
    "DINOv2": lambda: dino_processor,
    "ViT": lambda: vit_processor,
    "EfficientNetV2": lambda: None, # Non usa un processore Hugging Face separato per il training
    "ResNet50": lambda: None,
    "ConvNeXt": lambda: None,
}

# ====== CLASSE E FUNZIONE PER FINE-TUNING ======

class FineTunerModel(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes, model_identifier):
        super().__init__()
        self.base_model = base_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model_identifier = model_identifier # e.g., "CLIP", "ResNet50"
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Sblocca i parametri del modello base per il fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        # 'inputs' sono pixel_values per CLIP/DINO/ViT (già processati dal loro processor)
        # 'inputs' sono tensor di immagini trasformate per ResNet/EfficientNet/ConvNeXt
        features = None
        if self.model_identifier == "CLIP":
            features = self.base_model.get_image_features(pixel_values=inputs)
        elif self.model_identifier == "DINOv2":
            features = self.base_model(pixel_values=inputs).last_hidden_state[:, 0, :]
        elif self.model_identifier == "ViT":
            features = self.base_model(pixel_values=inputs).last_hidden_state[:, 0, :]
        elif self.model_identifier == "ResNet50":
            # Estrazione features per ResNet (fino a prima di fc)
            x = self.base_model.conv1(inputs)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = self.base_model.avgpool(x)
            features = torch.flatten(x, 1)
        elif self.model_identifier == "EfficientNetV2":
            # timm models typically have forward_features
            features = self.base_model.forward_features(inputs) # Output: [N, C, H, W]
            # Global average pooling
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        elif self.model_identifier == "ConvNeXt":
            x = self.base_model.features(inputs)
            x = self.base_model.avgpool(x)
            features = torch.flatten(x, 1)
        else:
            raise ValueError(f"Logica di forward non implementata per: {self.model_identifier}")

        return self.classifier(features)

def train_fine_tune_model(fine_tuner_model, dataloader, epochs, lr, base_lr, device, model_identifier, processor_fn=None):
    fine_tuner_model.to(device)
    
    # Parametri per l'ottimizzatore
    base_params = [p for n, p in fine_tuner_model.named_parameters() if "base_model" in n and p.requires_grad]
    classifier_params = [p for n, p in fine_tuner_model.named_parameters() if "classifier" in n and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': base_lr},
        {'params': classifier_params, 'lr': lr}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    print(f"\nInizio fine-tuning per {model_identifier} su {device} per {epochs} epoche...")
    for epoch in range(epochs):
        fine_tuner_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [{model_identifier}]")
        for images, labels in progress_bar:
            labels = labels.to(device)
            
            current_processor = processor_fn() if processor_fn else None

            if model_identifier in ["CLIP", "DINOv2", "ViT"] and current_processor:
                # 'images' sono tensor da ToTensor(), il processor gestisce la normalizzazione
                inputs_processed = current_processor(images=images, return_tensors="pt", padding=True, truncation=True, do_rescale=False).to(device)
                model_input = inputs_processed.pixel_values
            else:
                # Per ResNet, EfficientNet, ConvNeXt, 'images' sono già trasformate e normalizzate
                model_input = images.to(device)

            optimizer.zero_grad()
            outputs = fine_tuner_model(model_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_predictions)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 100. * correct_predictions / len(dataloader.dataset)
        print(f"Fine Epoch {epoch+1} [{model_identifier}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    fine_tuner_model.eval() # Metti il fine_tuner_model in eval mode dopo il training
    return fine_tuner_model # Ritorna il modello wrapper completo


# ====== ESTRAZIONE EMBEDDING ======
def l2_normalize(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10 
    return x / norm

def extract_clip_embeddings(image_paths):
    # clip_model qui è il base_model (CLIPModel) già fine-tuned o originale
    all_embeddings_list = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="CLIP Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb_batch = clip_model.get_image_features(**inputs) # Usa direttamente clip_model
        all_embeddings_list.append(emb_batch.cpu())
    concatenated_embeddings = torch.cat(all_embeddings_list, dim=0)
    return l2_normalize(concatenated_embeddings.numpy().astype("float32"))

def extract_dino_embeddings(image_paths):
    # dino_model qui è il base_model (AutoModel) già fine-tuned o originale
    all_embeddings_list = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="DINOv2 Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = dino_processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            emb_batch = outputs.last_hidden_state[:, 0, :] # CLS token
        all_embeddings_list.append(emb_batch.cpu())
    concatenated_embeddings = torch.cat(all_embeddings_list, dim=0)
    return l2_normalize(concatenated_embeddings.numpy().astype("float32"))

def extract_efficientnet_embeddings(image_paths):
    # efficientnet_model è il modello timm, già fine-tuned o originale
    embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="EfficientNetV2 Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = [transform_eff(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = efficientnet_model.forward_features(batch)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1)
            feats = feats.squeeze(-1).squeeze(-1)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

def extract_resnet_embeddings(image_paths):
    # resnet_model è il modello torchvision, già fine-tuned o originale
    # Il feature_extractor viene creato dal resnet_model corrente
    feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1]) # Rimuove il classificatore originale
    feature_extractor.eval().to(device) # Assicurati che sia in eval mode e sul device
    embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="ResNet50 Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = [transform_resnet(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch)
            feats = feats.view(feats.size(0), -1)
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

def extract_convnext_embeddings(image_paths):
    # convnext_model è il modello torchvision, già fine-tuned o originale
    # Il feature_extractor viene creato dal convnext_model corrente
    feature_extractor_convnext = nn.Sequential(convnext_model.features, convnext_model.avgpool)
    feature_extractor_convnext.eval().to(device)
    embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="ConvNeXt Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        # preprocess_convnext è definito globalmente durante l'init del modello
        imgs = [preprocess_convnext(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = feature_extractor_convnext(batch)
            feats = feats.view(feats.size(0), -1) # Flatten
        embeddings.append(feats.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

def extract_vit_embeddings(image_paths):
    # vit_model è il base_model (AutoModel) già fine-tuned o originale
    embeddings = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="ViT Embeddings"):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = vit_processor(images=batch_pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :] # CLS token
        embeddings.append(emb.cpu())
    return l2_normalize(torch.cat(embeddings, dim=0).numpy().astype("float32"))

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
    similarity = np.dot(query_embs, gallery_embs.T)
    scores = (similarity + 1) / 2 
    return scores

def retrieve_with_weighted_ensemble(query_files, gallery_files, model_scores_list, model_accuracies, model_names, k=50):
    num_queries = len(query_files)
    results = []
    
    accuracies_array = np.array(model_accuracies)
    if sum(accuracies_array) == 0 or len(accuracies_array) ==0 :
        weights = np.ones(len(model_scores_list)) / len(model_scores_list) if len(model_scores_list) > 0 else []
        print("Attenzione: tutte le accuratezze fornite sono 0 o non ci sono modelli. Si utilizzano pesi uguali (se possibile).")
    else:
        weights = accuracies_array / sum(accuracies_array)

    if weights.size > 0:
        print(f"Pesi dei modelli per la media ponderata (basati su {len(model_names)} modelli selezionati):")
        for name, weight, accuracy in zip(model_names, weights, model_accuracies):
            print(f" - {name}: {weight:.4f} (accuratezza fornita: {accuracy:.2f}%)")
    
    for q_idx in range(num_queries):
        query_path = query_files[q_idx]
        query_rel = query_path.replace("\\", "/")
        
        if not model_scores_list: # No scores to process
            results.append({"filename": query_rel, "gallery_images": []})
            continue

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
    query_folder = "testing_images5/test/query" 
    gallery_folder = "testing_images5/test/gallery" 

    ENABLED_MODELS = ["CLIP", "DINOv2", "EfficientNetV2", "ResNet50", "ConvNeXt", "ViT"]
    # ENABLED_MODELS = ["ResNet50", "EfficientNetV2", "ConvNeXt"]
    # ENABLED_MODELS = ["CLIP", "DINOv2", "EfficientNetV2", "ResNet50", "ConvNeXt", "ViT"]
    
    # --- DEFINISCI QUI LA MAPPATURA GLOBALE ---
    # Mappa i nomi "display" dei modelli ai nomi delle loro variabili globali effettive
    MODEL_GLOBAL_VAR_NAMES = {
        "CLIP": "clip_model",
        "DINOv2": "dino_model", # Questo è il nome della variabile globale
        "EfficientNetV2": "efficientnet_model",
        "ResNet50": "resnet_model",
        "ConvNeXt": "convnext_model",
        "ViT": "vit_model",
    }
    # --------------------------------------------

    MODEL_DEFAULT_ACCURACIES = {
        "CLIP": 50, "DINOv2": 50, "EfficientNetV2": 50,
        "ResNet50": 50, "ConvNeXt": 50, "ViT": 50,
    }

    for model_name in ENABLED_MODELS:
        if model_name not in MODEL_INITIALIZERS or model_name not in MODEL_EXTRACTORS:
            raise ValueError(f"Modello '{model_name}' non configurato correttamente.")
        if model_name not in MODEL_DEFAULT_ACCURACIES:
            print(f"Attenzione: Accuratezza di default non specificata per '{model_name}'. Verrà usato 0.0.")
            MODEL_DEFAULT_ACCURACIES[model_name] = 0.0

    if not ENABLED_MODELS:
        print("Nessun modello selezionato in ENABLED_MODELS. Uscita.")
        exit()
    
    print(f"Modelli selezionati: {', '.join(ENABLED_MODELS)}")

    # --- Inizializzazione di tutti i modelli abilitati (caricamento pesi pre-addestrati) ---
    for model_name in ENABLED_MODELS:
        MODEL_INITIALIZERS[model_name]() # Carica clip_model, dino_model, etc.

    # --- Fine-Tuning (se abilitato) ---
    if FINETUNING_CONFIG["ENABLED"] and FINETUNING_CONFIG["TRAIN_DATA_DIR"]:
        print(f"\n===== INIZIO FASE DI FINE-TUNING =====")
        if not os.path.exists(FINETUNING_CONFIG["TRAIN_DATA_DIR"]):
            print(f"ERRORE: La cartella di training specificata non esiste: {FINETUNING_CONFIG['TRAIN_DATA_DIR']}")
            print("Fine-tuning saltato.")
        else:
            # Determina NUM_CLASSES dal dataset di training
            # Usa una transform generica minima se quella specifica del modello non è ancora definita
            # (le transform specifiche vengono definite in init_*_model)
            temp_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            try:
                # Per caricare il dataset e ottenere il numero di classi, le trasformazioni devono essere definite.
                # Assicuriamoci che i modelli siano inizializzati per avere le transform pronte.
                # Questa è una sorta di pre-inizializzazione solo per ottenere le info del dataset.
                for model_name in ENABLED_MODELS:
                    if model_name in FINETUNING_CONFIG["MODELS_TO_FINETUNE"] and FINETUNING_CONFIG["MODELS_TO_FINETUNE"][model_name]:
                        if MODEL_TRANSFORMS_FOR_TRAINING[model_name] is None: # Se non ancora definito da init
                           MODEL_INITIALIZERS[model_name]() # Chiamata per definire la transform
                
                # Scegli una transform qualsiasi, solo per caricare il dataset e contare le classi
                # La transform corretta per ogni modello verrà usata durante il training effettivo
                representative_transform_key = next((m for m in ENABLED_MODELS if MODEL_TRANSFORMS_FOR_TRAINING.get(m) is not None), None)
                if representative_transform_key:
                    chosen_transform_for_dataset_check = MODEL_TRANSFORMS_FOR_TRAINING[representative_transform_key]
                else: # Fallback se nessun modello ha ancora una transform (improbabile dopo il loop sopra)
                    print("Attenzione: nessuna transform definita per il check del dataset, uso una generica.")
                    chosen_transform_for_dataset_check = temp_transform

                train_dataset_check = datasets.ImageFolder(FINETUNING_CONFIG["TRAIN_DATA_DIR"], transform=chosen_transform_for_dataset_check)
                NUM_CLASSES = len(train_dataset_check.classes)
                print(f"Trovate {NUM_CLASSES} classi nel dataset di training: {train_dataset_check.classes}")

                if NUM_CLASSES == 0:
                    print("ERRORE: Nessuna classe trovata nel dataset di training. Fine-tuning saltato.")
                else:
                    for model_name in ENABLED_MODELS:
                        if FINETUNING_CONFIG["MODELS_TO_FINETUNE"].get(model_name, False):
                            print(f"\n--- Fine-tuning per {model_name} ---")
                            # Mappa i nomi delle chiavi nel dizionario ai nomi delle variabili globali
                            model_variable_name_map = {
                                "CLIP": "clip_model",
                                "DINOv2": "dino_model", # Qui sta il problema! Non "dinov2_model"
                                "EfficientNetV2": "efficientnet_model",
                                "ResNet50": "resnet_model",
                                "ConvNeXt": "convnext_model",
                                "ViT": "vit_model",
                            }
                            base_model_instance = globals()[model_variable_name_map[model_name]]
                            
                            if base_model_instance is None:
                                print(f"Modello base {model_name} non inizializzato. Salto fine-tuning.")
                                continue

                            feature_dim = MODEL_EMBED_DIMS[model_name]
                            
                            # Prepara il DataLoader con le transform specifiche del modello per il training
                            current_train_transform = MODEL_TRANSFORMS_FOR_TRAINING[model_name]
                            if current_train_transform is None:
                                print(f"Attenzione: Transform di training non definite per {model_name}. Uso una generica.")
                                # Se si usa il processor (CLIP, DINO, ViT), ToTensor è sufficiente qui, il resto nel loop.
                                current_train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
                                if model_name in ["CLIP", "DINOv2", "ViT"]:
                                     # Per questi modelli, il processore gestisce la normalizzazione.
                                     # Assicurati che le dimensioni siano corrette.
                                     proc_size = (224,224) # default
                                     if model_name == "CLIP" and clip_processor: proc_size = (clip_processor.image_processor.size["height"], clip_processor.image_processor.size["width"])
                                     elif model_name == "DINOv2" and dino_processor: proc_size = (dino_processor.size["height"], dino_processor.size["width"])
                                     elif model_name == "ViT" and vit_processor: proc_size = (vit_processor.size["height"], vit_processor.size["width"])
                                     current_train_transform = transforms.Compose([
                                         transforms.Resize(proc_size), transforms.CenterCrop(proc_size), transforms.ToTensor()
                                     ])


                            train_dataset = datasets.ImageFolder(FINETUNING_CONFIG["TRAIN_DATA_DIR"], transform=current_train_transform)
                            train_loader = DataLoader(train_dataset, batch_size=FINETUNING_CONFIG["FINETUNE_BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)

                            fine_tuner = FineTunerModel(base_model_instance, feature_dim, NUM_CLASSES, model_name)
                            
                            # Recupera il processor specifico del modello, se esiste
                            model_processor_fn = MODEL_PROCESSORS.get(model_name)


                            trained_fine_tuner = train_fine_tune_model(
                                fine_tuner, train_loader,
                                FINETUNING_CONFIG["NUM_EPOCHS"],
                                FINETUNING_CONFIG["LEARNING_RATE"],
                                FINETUNING_CONFIG["BASE_MODEL_LEARNING_RATE"],
                                device, model_name, model_processor_fn
                            )
                            
                            # Aggiorna il modello globale con la base fine-tuned
                            # e mettilo in modalità valutazione e sul device corretto
                            globals()[model_variable_name_map[model_name]] = trained_fine_tuner.base_model.eval().to(device)
                            print(f"Fine-tuning completato per {model_name}. Modello base aggiornato.")
            except Exception as e:
                print(f"Errore durante la preparazione del dataset di training o il fine-tuning per un modello:")
                traceback.print_exc() # Stampa lo stack trace completo
                print("Fine-tuning saltato per questo modello.")
        print(f"===== FINE FASE DI FINE-TUNING =====\n")
    else:
        if not FINETUNING_CONFIG["ENABLED"]:
            print("Fine-tuning disabilitato globalmente.")
        elif not FINETUNING_CONFIG["TRAIN_DATA_DIR"]:
            print("Percorso al dataset di training non specificato. Fine-tuning saltato.")


    # --- Estrazione Embeddings ---
    query_files = sorted([os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    gallery_files = sorted([os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    if not query_files or not gallery_files:
        print(f"Errore: cartella query ({query_folder}) o gallery ({gallery_folder}) vuota o senza immagini supportate.")
        exit()
    
    print(f"Trovate {len(query_files)} immagini query e {len(gallery_files)} immagini gallery.")

    active_query_embeddings = {}
    active_gallery_embeddings = {}
    
    print("\n🔍 Estrazione degli embedding per i modelli selezionati...")
    for model_name in ENABLED_MODELS:
        # Assicurati che il modello (potenzialmente fine-tuned) sia in eval mode
        # Assicurati che MODEL_GLOBAL_VAR_NAMES sia accessibile qui (es. definito fuori dai blocchi if)
        current_model_instance = globals()[MODEL_GLOBAL_VAR_NAMES[model_name]]
        if current_model_instance:
            current_model_instance.eval() 
        else:
            print(f"Attenzione: Modello {model_name} non trovato per l'estrazione. Potrebbe non essere stato inizializzato.")
            continue


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
        if model_name not in active_query_embeddings or model_name not in active_gallery_embeddings:
            print(f"Embedding per {model_name} non trovati. Salto calcolo similarità.")
            continue
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
        query_files, gallery_files, 
        active_model_scores, 
        active_model_accuracies_for_ensemble,
        active_model_names_for_ensemble, 
        k=50
    )

    output_filename = "submission/ensemble_finetuned_t5_final.json" 
    save_submission(submission, output_filename)
    print(f"✅ Submission salvata in: {output_filename}")