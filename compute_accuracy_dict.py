import importlib.util
import os

# ✅ Percorso al file Python con il dizionario dei risultati
submission_file = "submission/submission_resnet50_ft_t5.py"

# ✅ Carica dinamicamente il modulo dal file Python
spec = importlib.util.spec_from_file_location("submission_results", submission_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# ✅ Accedi al dizionario "results"
data = module.data

total_correct = 0
total_checked = 0

for query_filename, gallery_filenames in data.items():
    # Estrai la classe dalla query (es. 'tram' da 'tram_query.jpg')
    query_class = query_filename.split('_')[0]

    # Conta le immagini nella gallery che contengono la classe nel nome
    correct = sum(query_class in img_name for img_name in gallery_filenames)

    print(f"Query '{query_class}': {correct}/{len(gallery_filenames)} immagini corrette")

    total_correct += correct
    total_checked += len(gallery_filenames)

# ✅ Statistiche globali
print("\nTotale immagini corrette:", total_correct)
print("Totale immagini verificate:", total_checked)
print("Accuratezza globale:", round(total_correct / total_checked * 100, 2), "%")
