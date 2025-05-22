# ML-project

Group members: Elisa Negrini, Michele Lovato Menin, Tommaso Ballarini

DUBBI 

- quante immagini ci danno per il mini training là?
- score di similarità conta solo se becchiamo tutte le anatre, o se becchiamo anche l’anatra più simile
- ci danno il fine tuning per le immagini del test?
- quanto grande la gallery

Per Zio Pietro
- Fine-tuning, non ha senso?
- Ha senso usare i top modelli e poi fare fine tuning sul nostro dominio o è ci sono altre strategie?
- Usare git-hub, come non creare conflitti
- Perchè usare la azure machine? o basta colab?

Possibili problematiche:
La funzione get_feature_extractor processa le immagini una alla volta. Questo può essere lento per gallery e query di grandi dimensioni.
Suggerimento: Modifica get_feature_extractor per processare le immagini in batch.
se abbiamo una gallery gigantesca ci mettiamo tabto a estrarre l'embedding per le immagini se le estraiamo una ad una


RESULTS:
- modello_dino.py             ----- t7 ---- 95.92% -- k = 50
- modello_minestrone_ CDER.py ----- t7 ---- 94.48% -- k = 50
- model_efficient_net_v2_l.py ----- t7 ---- 91.73% -- k = 30
- model_efficient_net_v2_l.py ----- t7 ---- 90.48% -- k = 50
- model_clip_vit_base_..ipynb ----- t7 ---- 79.51% -- k = 49
- model_resnet50.py           ----- t7 ---- 74.96% -- k = 50

- model_efficient_net_v2_l.py ----- t6 ---- 54.42% -- k = 50
- modello_minestrone_ CDER.py ----- t6 ---- 48.42% -- k = 50
- model_clip_vit_base_..ipynb ----- t6 ---- 56.71% -- k = 49
- modello_dino.py             ----- t6 ---- 43.16% -- k = 50
- model_efficient_net_v2_l.py ----- t6 ---- 59.12% -- k = 30
- model_clip_vit_base_..ipynb ----- t6 ---- 66.75% -- k = 20

- model_efficient_net_v2_l.py ----- t4 ---- 68.33% -- k = 30 
- modello_dino.py             ----- t4 ---- 61.11% -- k = 30
- model_efficient_net_v2_l.py ----- t4 ---- 42.00% -- k = 50 
- modello_dino.py             ----- t4 ---- 39.33% -- k = 50

- modello_dino.py             ----- t1 ---- 98.33% -- k = 30
- modello_dino.py             ----- t1 ---- 97.75% -- k = 50
- model_resnet50.py           ----- t1 ---- 84.58% -- k = 30
- model_clip_vit_base_..ipynb ----- t1 ---- 82.14% -- k = 49
