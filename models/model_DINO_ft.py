import torch
import torch.nn as nn
from torchvision import datasets # Rimosso transforms perché il processore gestisce tutto
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from transformers import AutoImageProcessor, AutoModel # Modificato per DINOv2
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica DINOv2 e il suo Image Processor
MODEL_NAME = "facebook/dinov2-base"
try:
    dino_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Errore durante il caricamento del modello {MODEL_NAME}: {e}")
    print("Assicurati di avere una connessione internet e che il nome del modello sia corretto.")
    print("Potrebbe essere necessario installare/aggiornare transformers: pip install transformers --upgrade")
    # Esci o gestisci l'errore come preferisci se il modello non può essere caricato
    exit()

# Classe per il fine-tuning che permette di aggiornare tutti i parametri
class ImageClassifierFineTuner(nn.Module):
    def __init__(self, base_model, embed_dim, num_classes, unfreeze_layers=True):
        super().__init__()
        self.base_model = base_model
        # Il classificatore prende in input le feature estratte dal modello base
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Se vogliamo fare fine-tuning, sblocchiamo i parametri del modello base
        if unfreeze_layers:
            for param in self.base_model.parameters(): # Sblocca tutti i parametri del base_model
                param.requires_grad = True
        else:
            # Altrimenti, congela il modello base (solo feature extraction)
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Assicurati che il classificatore sia sempre addestrabile
        for param in self.classifier.parameters():
            param.requires_grad = True


    def forward(self, pixel_values):
        # Estrae le feature dal modello base.
        # Per DINOv2, usiamo l'output dell'ultimo strato nascosto, prendendo il token [CLS]
        # Non usiamo torch.no_grad() qui per permettere il backpropagation attraverso il base_model se sbloccato
        outputs = self.base_model(pixel_values=pixel_values)
        # features = outputs.pooler_output # Alcuni modelli hanno pooler_output
        features = outputs.last_hidden_state[:, 0, :] # Prende l'embedding del token [CLS]
        return self.classifier(features)

def custom_collate(batch):
    # Filter out PIL Images and collate the rest
    tensors, labels = [], []
    for image, label in batch:
        tensors.append(image)  # Store PIL Images separately
        labels.append(label)

    # Collate labels using default collation
    labels = default_collate(labels)

    return tensors, labels  # Return PIL Images and collated labels

def train_model(model, dataloader, image_processor_func, epochs=5, lr=5e-5):
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Creiamo due gruppi di parametri con learning rate diversi
    # Filtra i parametri che effettivamente richiedono gradiente
    base_params = [p for n, p in model.named_parameters() if "base_model" in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad]

    # Controlla se ci sono parametri da ottimizzare nel base_model
    optimizer_params = []
    if base_params:
        optimizer_params.append({'params': base_params, 'lr': lr / 10}) # LR più basso per il modello pre-addestrato
    if classifier_params:
        optimizer_params.append({'params': classifier_params, 'lr': lr}) # LR standard per il classificatore
    else: # Se solo il base_model è sbloccato (improbabile ma per sicurezza)
        if not base_params:
             print("Attenzione: Nessun parametro da addestrare!")
             return model # Non c'è nulla da addestrare

    optimizer = torch.optim.AdamW(optimizer_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for pil_images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Le immagini dal DataLoader sono ora PIL Images
            labels = labels.to(device)

            # Preprocessing con l'image_processor specifico del modello (es. DINOv2)
            # image_processor_func gestisce la conversione a tensori, resize, crop, normalizzazione
            try:
                inputs = image_processor_func(images=pil_images, return_tensors="pt", padding=True, truncation=True).to(device)
            except Exception as e:
                print(f"Errore durante il processing delle immagini: {e}")
                # Potresti voler ispezionare una delle pil_images qui
                # print(f"Tipo immagine: {type(pil_images[0]) if pil_images else 'N/A'}")
                continue # Salta questo batch

            optimizer.zero_grad()
            outputs = model(inputs.pixel_values) # Passa i pixel_values processati
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Calcolo dell'accuratezza
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
        epoch_acc = (100.0 * correct / total) if total > 0 else 0
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return model

def get_feature_extractor(trained_model, image_processor_func):
    # trained_model è l'istanza di ImageClassifierFineTuner dopo l'addestramento
    # Vogliamo estrarre le feature dal *base_model* al suo interno
    feature_extractor_model = trained_model.base_model
    feature_extractor_model.eval() # Assicuriamoci che il modello base sia in modalità valutazione

    def extractor(image_paths):
        embs = []
        for path in tqdm(image_paths, desc="Extracting features"):
            try:
                image = Image.open(path).convert("RGB")
                # Processa l'immagine usando l'image_processor
                inputs = image_processor_func(images=image, return_tensors="pt").to(device)
            except FileNotFoundError:
                print(f"Attenzione: File immagine non trovato {path}")
                continue
            except Exception as e:
                print(f"Errore nell'aprire o processare l'immagine {path}: {e}")
                continue


            with torch.no_grad():
                # Estrae le feature usando il base_model (es. DINOv2)
                outputs = feature_extractor_model(pixel_values=inputs.pixel_values)
                # emb = outputs.pooler_output # Se DINOv2 avesse un pooler_output significativo per questo task
                emb = outputs.last_hidden_state[:, 0, :] # Prende l'embedding del token [CLS]
                embs.append(emb.cpu().numpy()[0])

        if not embs: # Se nessuna embedding è stata estratta (es. tutti i file mancanti)
            return np.array([]).astype("float32") # Restituisce un array vuoto con il tipo corretto
        return np.array(embs).astype("float32")
    return extractor

def retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=10):
    if query_embs.shape[0] == 0 or gallery_embs.shape[0] == 0:
        print("Attenzione: Non ci sono embedding per query o gallery. Impossibile eseguire il retrieval.")
        return []

    model_nn = NearestNeighbors(n_neighbors=min(k, gallery_embs.shape[0]), metric='cosine') # k non può essere > num_samples
    model_nn.fit(gallery_embs)
    distances, indices = model_nn.kneighbors(query_embs)

    results = []
    for i, query_path in enumerate(query_files):
        query_rel = query_path.replace("\\", "/")
        gallery_matches = [gallery_files[idx].replace("\\", "/") for idx in indices[i]]
        results.append({
            "filename": query_rel,
            "gallery_images": gallery_matches
        })
    return results

def save_submission(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

# ===============================
# ESECUZIONE COMPLETA
# ===============================

# Step 1: Prepara il dataset. ImageFolder caricherà le immagini come PIL.
# Nessuna trasformazione specificata qui, così ImageFolder restituisce immagini PIL.
# Le trasformazioni (resize, crop, normalizzazione) saranno gestite da image_processor.
try:
    train_dataset = datasets.ImageFolder("testing_images5/training") # Carica PIL Images
    if not train_dataset.classes:
        print("Errore: Nessuna classe trovata nel dataset di training. Controlla il percorso e la struttura della cartella.")
        exit()
    num_classes = len(train_dataset.classes)
    print(f"Trovate {num_classes} classi: {train_dataset.classes}")
except FileNotFoundError:
    print("Errore: Cartella di training 'testing_images5/training' non trovata.")
    print("Assicurati che il percorso sia corretto e che Google Drive sia montato se usi Colab.")
    exit()
except Exception as e:
    print(f"Errore durante il caricamento del dataset di training: {e}")
    exit()


# DataLoader ora gestirà batch di immagini PIL e labels
# Il collate_fn di default dovrebbe funzionare bene, raggruppando le immagini PIL in una lista.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate) # Utilizzo della funzione custom_collate

# Step 2: Fine-tune DINOv2
# La dimensione dell'embedding per dinov2-base è 768
embed_dim = dino_model.config.hidden_size # Più robusto che hardcodare 768
model_to_train = ImageClassifierFineTuner(dino_model, embed_dim=embed_dim, num_classes=num_classes, unfreeze_layers=True)

print("Inizio fine-tuning del modello...")
# Passiamo image_processor alla funzione train_model per processare i batch di immagini PIL
# Aumentato numero di epoche, LR potrebbe necessitare di tuning per DINOv2
trained_fine_tuned_model = train_model(model_to_train, train_loader, image_processor_func=image_processor, epochs=10, lr=1e-5) # LR ridotto per DINOv2

# Step 3: Estrai features da query e gallery usando il modello fine-tuned
# Passiamo il modello fine-tuned completo e l'image_processor
# get_feature_extractor accederà a trained_fine_tuned_model.base_model internamente
feature_extractor_fn = get_feature_extractor(trained_fine_tuned_model, image_processor_func=image_processor)

query_folder = "testing_images5/test/query"
gallery_folder = "testing_images5/test/gallery"

try:
    query_files = [os.path.join(query_folder, fname) for fname in os.listdir(query_folder) if fname.lower().endswith((".jpg", ".jpeg", ".png"))]
    gallery_files = [os.path.join(gallery_folder, fname) for fname in os.listdir(gallery_folder) if fname.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not query_files:
        print(f"Attenzione: Nessun file immagine trovato in {query_folder}")
    if not gallery_files:
        print(f"Attenzione: Nessun file immagine trovato in {gallery_folder}")

except FileNotFoundError as e:
    print(f"Errore: Cartella query o gallery non trovata. Controlla i percorsi: {e}")
    exit()
except Exception as e:
    print(f"Errore durante la lettura dei file query/gallery: {e}")
    exit()

if not query_files or not gallery_files:
    print("Nessun file query o gallery da processare. Uscita.")
    exit()

print(f"Trovati {len(query_files)} file query e {len(gallery_files)} file gallery.")

query_embs = feature_extractor_fn(query_files)
gallery_embs = feature_extractor_fn(gallery_files)

# Step 4: Retrieval
if query_embs.shape[0] > 0 and gallery_embs.shape[0] > 0:
    submission = retrieve_query_vs_gallery(query_embs, query_files, gallery_embs, gallery_files, k=50)   # <--- MODIFICARE QUESTO K
    # Step 5: Salvataggio
    submission_path = "submission/submission_dino_ft_t5.json" # Nome file modificato
    save_submission(submission, submission_path)
    print(f"✅ Submission salvata in: {submission_path}")
else:
    print("Nessuna embedding estratta, impossibile eseguire il retrieval o salvare la submission.")

print("Esecuzione completata.")
