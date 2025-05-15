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
