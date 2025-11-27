# IDS-with-1D-CNN

Questo progetto implementa un sistema di Intrusion Detection (IDS) basato su una rete neurale convoluzionale 1D (CNN) per la classificazione di traffico di rete come normale o anomalo.

## Struttura del progetto
- **src/network.py**: Definisce la classe `Dataset` per la preparazione dei dati e la classe `CNN` per il modello.
- **src/train.py**: Gestisce il training del modello, la validazione e il logging.
- **src/test.py**: Esegue la valutazione del modello su dati di test.
- **src/heatmap.py**: Analizza la correlazione tra le features e la label, mostrando le più informative.
- **Dataset/**: Contiene i file CSV dei dati e delle features.
- **requirements.txt**: Elenco delle librerie Python necessarie.
- **run.sh**: Script per la creazione dell'ambiente virtuale e l'avvio del training/test.
- **main.py**: Entry point per lanciare il training o il test tramite parametri da linea di comando.

## Funzionamento
1. **Preparazione ambiente**: Esegui `./run.sh train` per creare la virtualenv e installare le dipendenze.
2. **Training**: Il modello viene addestrato su un dataset bilanciato (selezione casuale di record con label 1 e 0). Le features più informative vengono selezionate tramite correlazione.
3. **Validazione**: I dati non usati per il training vengono usati per la validazione, garantendo che train e validation siano disgiunti.
4. **Test**: Puoi eseguire la valutazione del modello su nuovi dati con `./run.sh test`.
5. **Analisi features**: `src/heatmap.py` genera una heatmap delle correlazioni tra features e label per aiutare la selezione delle variabili più utili.

## Parametri principali
- `--epochs`: Numero di epoche di training
- `--batch_size`: Dimensione dei batch
- `--learning_rate`: Learning rate dell'ottimizzatore
- `--csv_path`: Percorso del dataset
- `--model_path`: Percorso dove salvare/caricare il modello

## Note
- Il progetto è ottimizzato per chip Apple Silicon (M1/M2) e supporta device `mps`.
- Il dataset viene preprocessato per evitare problemi di memoria e overfitting.
- Tutti i log e i risultati vengono salvati in file dedicati.

## Esempio di utilizzo
```bash
./run.sh train --epochs 20 --batch_size 128 --csv_path ./Dataset/UNSW-NB15_1.csv
./run.sh test --csv_path ./Dataset/UNSW-NB15_1.csv --model_path cnn_model.pth
```