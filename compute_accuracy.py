import json

# 🔁 Modifica questo percorso con quello corretto del tuo file JSON
json_path = "submission/submission_dino_t1.json"

# Caricamento dei dati dal file JSON
with open(json_path, "r") as f:
    data = json.load(f)

total_correct = 0
total_checked = 0

for entry in data:
    query_path = entry["filename"]
    gallery_paths = entry["gallery_images"]

    # Classe della query (es. 'tram' da 'tram_query.jpg')
    query_class = query_path.split('/')[-1].split('_')[0]

    # Conta quante immagini nella gallery contengono la classe nel nome
    correct = sum(query_class in img_path for img_path in gallery_paths)

    print(f"Query '{query_class}': {correct}/{len(gallery_paths)} immagini corrette")

    total_correct += correct
    total_checked += len(gallery_paths)

# Statistiche globali basate solo sulle immagini effettivamente verificate
print("\nTotale immagini corrette:", total_correct)
print("Totale immagini verificate:", total_checked)
print("Accuratezza globale:", round(total_correct / total_checked * 100, 2), "%")