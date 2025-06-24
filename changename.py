import os

def rinomina_immagini(cartella, nome_animale):
    # Lista tutti i file nella cartella
    immagini = [f for f in os.listdir(cartella) if f.lower().endswith(".jpeg")]
    immagini.sort()  # ordina alfabeticamente per consistenza

    for i, nome_vecchio in enumerate(immagini, start=1):
        estensione = os.path.splitext(nome_vecchio)[1]
        nuovo_nome = f"{nome_animale}_{i:02d}{estensione}"
        percorso_vecchio = os.path.join(cartella, nome_vecchio)
        percorso_nuovo = os.path.join(cartella, nuovo_nome)

        os.rename(percorso_vecchio, percorso_nuovo)
        print(f"Rinominato: {nome_vecchio} -> {nuovo_nome}")

# Esempio di utilizzo:
# cambia "path/alla/cartella" con il percorso reale
# e "antilope" con il nome desiderato
if __name__ == "__main__":
    rinomina_immagini("ML-project/testing_images8_animals/test/gallery/antelope", "antelope")
