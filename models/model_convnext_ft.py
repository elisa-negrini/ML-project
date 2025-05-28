import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import json
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===============================
# CONFIGURAZIONE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Percorsi per le immagini
# NOTA: Assicurati che questa cartella esista e contenga sottocartelle per ogni classe se fai fine-tuning
FINETUNE_TRAIN_FOLDER = "testing_images5/training" # Esempio: "dataset/train"
QUERY_FOLDER = "testing_images5/test/query"
GALLERY_FOLDER = "testing_images5/test/gallery"
OUTPUT_SUBMISSION_PATH = "submission/submission_convnext_finetuned_t5.py" # o .json

# Parametri per il Fine-tuning
DO_FINETUNING = True  # Imposta a True per eseguire il fine-tuning
NUM_EPOCHS_FINETUNE = 5 # Numero di epoche per il fine-tuning
LEARNING_RATE_FINETUNE = 1e-4
BATCH_SIZE_FINETUNE = 16
UNFREEZE_LAYERS_FINETUNE = True # True per sbloccare i layer del base_model, False per addestrare solo il classificatore

# ===============================
# CARICAMENTO MODELLO CONVNEXT PRE-ADDESTRATO E PREPROCESSING
# ===============================
weights = ConvNeXt_Base_Weights.DEFAULT
convnext_model = convnext_base(weights=weights).to(device)

# Definisci il preprocessing per ConvNeXt
# Questo preprocess sarà usato sia per il training (se DO_FINETUNING=True) sia per l'estrazione delle features
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # ConvNeXt si aspetta immagini 224x224
    transforms.ToTensor(),
    weights.transforms(),  # Utilizza direttamente le trasformazioni fornite dai pesi (normalizzazione)
])

# ===============================
# CLASSE PER IL FINE-TUNING DI CONVNEXT
# ===============================
class ConvNeXtFineTuner(nn.Module):
    def __init__(self, base_model, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model # Istanza del modello convnext_base

        # L'ultimo layer del classificatore di convnext_base è base_model.classifier[2]
        original_classifier_layer = self.base_model.classifier[2]
        embed_dim = original_classifier_layer.in_features # Per convnext_base è 1024

        # Sostituisci l'ultimo layer lineare con uno nuovo per il numero di classi del tuo dataset
        self.base_model.classifier[2] = nn.Linear(embed_dim, num_classes)

        if unfreeze_layers:
            print("Fine-tuning: Unfreezing all layers of the base model.")
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            print("Fine-tuning: Freezing base model layers, training only the new classifier.")
            # Congela tutti i layer tranne il nuovo classificatore
            for param in self.base_model.features.parameters():
                param.requires_grad = False
            # avgpool di solito non ha parametri allenabili
            # classifier[0] è LayerNorm2d, classifier[1] è Flatten
            for param in self.base_model.classifier[0].parameters():
                param.requires_grad = False
            # Il nuovo layer classificatore (classifier[2]) deve essere allenabile
            for param in self.base_model.classifier[2].parameters():
                param.requires_grad = True

    def forward(self, pixel_values):
        # pixel_values sono i tensori delle immagini preprocessati
        return self.base_model(pixel_values)

# ===============================
# FUNZIONE DI TRAINING
# ===============================
def train_model(model, dataloader, epochs=5, lr=5e-5):
    model = model.to(device)
    model.train() # Imposta il modello in modalità training
    criterion = nn.CrossEntropyLoss()

    # Separa i parametri del base_model (features, avgpool, classifier[0]) da quelli del nuovo classificatore (classifier[2])
    base_model_params = []
    # Features
    base_model_params.extend(filter(lambda p: p.requires_grad, model.base_model.features.parameters()))
    # AvgPool (se ha parametri e sono sbloccati)
    if hasattr(model.base_model, 'avgpool') and model.base_model.avgpool is not None:
        base_model_params.extend(filter(lambda p: p.requires_grad, model.base_model.avgpool.parameters()))
    # Classifier[0] (LayerNorm)
    base_model_params.extend(filter(lambda p: p.requires_grad, model.base_model.classifier[0].parameters()))

    classifier_head_params = list(filter(lambda p: p.requires_grad, model.base_model.classifier[2].parameters()))

    optimizer_params = []
    if base_model_params:
        optimizer_params.append({'params': base_model_params, 'lr': lr / 10}) # Learning rate più basso per il base_model
        print(f"Optimizer: Training {len(base_model_params)} parameters from base model with LR {lr/10}")
    if classifier_head_params:
        optimizer_params.append({'params': classifier_head_params, 'lr': lr})
        print(f"Optimizer: Training {len(classifier_head_params)} parameters from classifier head with LR {lr}")
    
    if not optimizer_params:
        print("Warning: No parameters to train. Check unfreeze_layers and model structure.")
        return model

    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) # model è ConvNeXtFineTuner, chiama il suo forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct_predictions / total_samples)

        epoch_loss = running_loss / total_samples
        epoch_acc = 100. * correct_predictions / total_samples
        print(f"Epoch {epoch+1} Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    model.eval() # Riporta il modello in modalità valutazione dopo il training
    return model

# ===============================
# FUNZIONI PER ESTRAZIONE FEATURES E RETRIEVAL
# ===============================

# Costruisce un modello per estrarre feature dal ConvNeXt (output prima del layer di classificazione finale)
def build_feature_extractor(convnext_model_instance):
    # L'istanza convnext_model_instance è il modello ConvNeXt (originale o fine-tuned)
    # Vogliamo le features prima del suo ultimo layer lineare (classifier[2])
    feature_extractor = nn.Sequential(
        convnext_model_instance.features,
        convnext_model_instance.avgpool,
        convnext_model_instance.classifier[0],  # LayerNorm2d
        convnext_model_instance.classifier[1]   # Flatten
    ).to(device).eval() # Metti in modalità valutazione
    return feature_extractor

# Funzione per estrarre le feature da una lista di immagini usando un dato estrattore
def extract_features(image_paths, feature_extractor_model, preprocess_transform):
    features_list = []
    for path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(path).convert("RGB")
            inputs = preprocess_transform(image).unsqueeze(0).to(device)  # Aggiungi batch dimension
            with torch.no_grad():
                feature_vec = feature_extractor_model(inputs)
            features_list.append(feature_vec.cpu().numpy()[0])  # Estrai feature come array numpy
        except Exception as e:
            print(f"Errore nell'elaborare l'immagine {path}: {e}")
            # Puoi decidere se aggiungere un vettore nullo o saltare l'immagine
            # features_list.append(np.zeros(shape_of_feature_vector, dtype="float32"))
    if not features_list:
        # Determina la dimensione attesa delle feature da feature_extractor_model se possibile
        # o restituisci un array vuoto con la forma corretta se conosciuta
        # Esempio: return np.array([]).reshape(0, 1024).astype("float32") per convnext_base
        # Per ora, se la lista è vuota, determiniamo la dimensione da un input fittizio
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                dummy_output = feature_extractor_model(dummy_input)
            feature_dim = dummy_output.shape[1]
            return np.array([]).reshape(0, feature_dim).astype("float32")
        except Exception as e:
            print(f"Impossibile determinare la dimensione delle feature, restituito array vuoto: {e}")
            return np.array([]).astype("float32")

    return np.array(features_list).astype("float32")

# Retrieval k-NN
def retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=5):
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Embedding di query o gallery vuoti. Impossibile eseguire il retrieval.")
        # Restituisci una struttura vuota coerente con l'output atteso
        results = []
        for query_path in query_files:
            query_rel = query_path.replace("\\", "/")
            results.append({
                "filename": query_rel,
                "gallery_images": []
            })
        return results

    model_knn = NearestNeighbors(n_neighbors=min(k, gallery_embs.shape[0]), metric='cosine', algorithm='brute')
    model_knn.fit(gallery_embs)
    distances, indices = model_knn.kneighbors(query_embs)

    results = []
    for i, query_path in enumerate(query_files):
        query_rel = query_path.replace("\\", "/")
        gallery_matches = [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
        results.append({
            "filename": query_rel,
            "gallery_images": gallery_matches
        })
    return results

# Salvataggio JSON
def save_submission_json(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def save_submission_py_dict(data_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("data = {\n")
        for i, (key, value) in enumerate(data_dict.items()):
            # Assicurati che i percorsi dei file siano stringhe correttamente formattate per Python
            formatted_value_list = [f'"{str(v)}"' for v in value]
            f.write(f'    "{str(key)}": [{", ".join(formatted_value_list)}]')
            if i < len(data_dict) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("}\n")

# ===============================
# ESECUZIONE
# ===============================
if __name__ == "__main__":
    # Modello ConvNeXt da usare (originale o fine-tuned)
    model_to_use_for_features = convnext_model

    if DO_FINETUNING:
        print(f"Inizio fase di Fine-tuning del modello ConvNeXt su: {FINETUNE_TRAIN_FOLDER}")
        if not os.path.isdir(FINETUNE_TRAIN_FOLDER):
            print(f"ERRORE: La cartella di training per il fine-tuning '{FINETUNE_TRAIN_FOLDER}' non esiste.")
            print("Imposta DO_FINETUNING = False o fornisci un percorso valido.")
            exit()

        # Carica il dataset di training per il fine-tuning
        try:
            train_dataset = datasets.ImageFolder(FINETUNE_TRAIN_FOLDER, transform=preprocess)
            if not train_dataset.classes:
                 print(f"ERRORE: Nessuna classe trovata in '{FINETUNE_TRAIN_FOLDER}'. Verifica la struttura della cartella.")
                 exit()
            print(f"Trovate {len(train_dataset.classes)} classi per il fine-tuning: {train_dataset.classes}")
        except Exception as e:
            print(f"Errore nel caricamento del dataset di fine-tuning: {e}")
            exit()

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_FINETUNE, shuffle=True, num_workers=2, pin_memory=True)
        num_classes_finetune = len(train_dataset.classes)

        # Istanzia il modello per il fine-tuning
        # Questo modifica convnext_model aggiungendo un nuovo classificatore
        convnext_finetuner_wrapper = ConvNeXtFineTuner(convnext_model, # Passa l'istanza caricata
                                                       num_classes=num_classes_finetune,
                                                       unfreeze_layers=UNFREEZE_LAYERS_FINETUNE)
        
        # Addestra il modello (convnext_finetuner_wrapper.base_model viene modificato in-place)
        trained_finetuner_model = train_model(convnext_finetuner_wrapper,
                                              train_loader,
                                              epochs=NUM_EPOCHS_FINETUNE,
                                              lr=LEARNING_RATE_FINETUNE)
        
        # Il modello addestrato è trained_finetuner_model.base_model
        model_to_use_for_features = trained_finetuner_model.base_model
        print("Fine-tuning completato.")
    else:
        print("Fine-tuning saltato. Si utilizzerà il modello ConvNeXt pre-addestrato per l'estrazione delle features.")

    # Costruisci l'estrattore di feature dal modello scelto (originale o fine-tuned)
    print("Costruzione dell'estrattore di features...")
    final_feature_extractor = build_feature_extractor(model_to_use_for_features)

    # Estrazione delle features per query e gallery
    print("Estrazione features per le immagini query...")
    query_files = [os.path.join(QUERY_FOLDER, f) for f in os.listdir(QUERY_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    gallery_files = [os.path.join(GALLERY_FOLDER, f) for f in os.listdir(GALLERY_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not query_files:
        print(f"ERRORE: Nessun file query trovato in {QUERY_FOLDER}")
        exit()
    if not gallery_files:
        print(f"ERRORE: Nessun file gallery trovato in {GALLERY_FOLDER}")
        exit()

    query_embs = extract_features(query_files, final_feature_extractor, preprocess)
    print("Estrazione features per le immagini gallery...")
    gallery_embs = extract_features(gallery_files, final_feature_extractor, preprocess)

    # Esegui il retrieval
    print("Esecuzione del retrieval k-NN...")
    submission_list = retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=10)

    # Prepara i dati per il salvataggio nel formato dizionario Python
    data= {
        os.path.basename(entry['filename']): [os.path.basename(img) for img in entry['gallery_images']]
        for entry in submission_list
    }
    
    print(data)
    # Salva la submission come dizionario Python (file .py)
    save_submission_py_dict(data, OUTPUT_SUBMISSION_PATH)
    print(f"✅ Submission salvata come dizionario Python in: {OUTPUT_SUBMISSION_PATH}")


    # Se vuoi anche il formato JSON:
    # submission_path_json = "submission/submission_convnext_t6_finetuned.json"
    # save_submission_json(submission_list, submission_path_json)
    # print(f"✅ Submission salvata come JSON in: {submission_path_json}")