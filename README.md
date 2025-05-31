# Estensione della procedura di training con torneo di funzioni di perdita e ensemble intelligente

Questo progetto estende la procedura di training tradizionale con due contributi principali:

1. **Weak pretraining come torneo tra funzioni di perdita**
2. **Voting finale tramite ensemble sistematico dei migliori modelli**

## 1. Weak Pretraining con Torneo tra Funzioni di Perdita

La procedura di pretraining è stata modificata per testare in parallelo diverse funzioni di perdita. Ogni `ModelTrainer` viene instanziato con una diversa loss tra quelle selezionate (es. `cross_entropy_loss`, `ncod_loss`, ecc.).

L’idea è trattare ogni loss come un partecipante a un torneo, eseguendo cicli di addestramento su tutto il dataset **ABCD**. Al termine di ogni round di training, vengono valutati i modelli e si seleziona, per ogni loss, quello con il miglior punteggio di validazione (F1 score). Successivamente, si seleziona la loss che ha ottenuto il **miglior risultato complessivo** in quel round.

**Nota:** al momento si seleziona solo la loss che ha performato meglio nell’ultimo round, ma sarebbe opportuno calcolare una media o statistica su tutti i round precedenti. Questo è stato lasciato come `TODO`.

## 2. Gestione dei Modelli Preaddestrati

Durante il training, se sono disponibili modelli già preaddestrati, questi vengono caricati automaticamente. Viene selezionato il **modello peggiore** (basato sul F1 score) per cercare di migliorarlo.

Se un modello con prestazioni migliori viene ottenuto, **sostituisce il peggiore** tra quelli preaddestrati. Questo meccanismo permette di mantenere nel tempo una collezione di modelli in costante miglioramento, limitando il numero di checkpoint salvati.

## 3. Ensemble dei Migliori Modelli

Una volta completato il training, viene eseguita una selezione dei modelli migliori in base al punteggio F1. In particolare:

- Viene caricato il metadata associato a ciascun modello.
- I modelli vengono ordinati per F1 score.
- Vengono selezionati i top-k modelli (default: `top_k = 5`).
- Si costruisce un ensemble utilizzando questi modelli per la procedura di voting finale.

Questa strategia supera l’ipotesi precedente secondo cui tutti i “migliori” modelli (uno per loss) siano effettivamente buoni. Ora si adotta una **classifica relativa**, includendo sistematicamente solo quelli realmente più performanti, migliorando la qualità del voting.

## Conclusioni

Queste modifiche rendono la procedura di training più robusta e adattiva:

- Le funzioni di perdita competono tra loro, selezionando dinamicamente la più adatta.
- I modelli vengono continuamente aggiornati e migliorati.
- Il voting si basa su un ensemble ottimizzato, non più su scelte arbitrarie o statiche.

Il sistema nel suo complesso è pensato per adattarsi in modo automatico e intelligente alla complessità del dataset e alle variazioni nei risultati tra diversi round di training.
