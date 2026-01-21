# FDS_IDS_CNN — Intrusion Detection Unsupervised (UNSW-NB15)

Questo progetto implementa un sistema di Intrusion Detection (IDS) in modalità unsupervised basato su due approcci principali:

- Autoencoder convoluzionale (ConvAE) per individuare anomalie tramite errore di ricostruzione
- Isolation Forest per l’anomalia detection su feature scalate

I dati sono derivati dal dataset UNSW-NB15 e vengono preparati con uno script di estrazione che seleziona un sottinsieme focalizzato sui top 5 attacchi e crea split coerenti per addestramento ed evaluation.

Per la descrizione completa dell’obiettivo, delle scelte progettuali e dei risultati sperimentali, fai riferimento al report: Report.pdf.

## Struttura del progetto

- Dataset/: CSV originali e i file generati per train/eval
- src/: codice sorgente (modello, training, testing, utils, data prep)
- Model/: modelli e soglie salvati (AE e IF, più scaler)
- Plots/: grafici di training e distribuzioni degli score
- Logs/: log di esecuzione con metriche e confusion matrix
- run_unsup.py: entrypoint unico per training e testing (AE/Isolation Forest)
- requirements.txt: dipendenze Python

## Dati e preparazione

Lo script di preparazione dati unifica i file UNSW-NB15 e produce due CSV:

- `Dataset/UNSW_top5_train.csv`: set di training (feature numeriche, senza colonne testuali e senza label)
- `Dataset/UNSW_top5_eval.csv`: set di evaluation (feature numeriche + `Label` e metadati necessari)

La procedura è automatizzabile direttamente dal runner oppure richiamando il modulo di estrazione.

Esecuzione (opzione comoda dal runner):

```bash
python run_unsup.py --mode train --type ae --build_dataset
```

Oppure manualmente:

```bash
python -c "from src.data_extraction import build_datasets; build_datasets()"
```

Dettagli operativi sul parsing dei CSV:

- Rimozione di colonne non numeriche comuni (ad es. `srcip`, `dstip`, `proto`, `service`, `attack_cat`, `state`)
- Conversione robusta a numerico con `errors='coerce'` e `fillna(0)`
- Applicazione di `StandardScaler` quando richiesto

## Metodi

### Autoencoder Convoluzionale (ConvAE)

- Architettura 1D con `Conv1d` + `BatchNorm` + `MaxPool`, bottleneck lineare e decoder con `ConvTranspose1d`
- Allenamento su feature scalate con loss MSE
- Soglia di anomalia definita come 95° percentile dell’errore di ricostruzione sul training
- Valutazione su `UNSW_top5_eval.csv` con metriche (`accuracy`, `classification_report`, `confusion_matrix`)

Output salvati:

- `Model/conv_autoencoder.pth`: pesi del modello
- `Model/recon_threshold.json`: soglia di errore di ricostruzione
- `Model/scaler.pkl`: scaler usato per il preprocessing
- `Plots/unsup_loss.png`: andamento della loss di training

### Isolation Forest (IF)

- Preprocessing con `StandardScaler` su feature (fit preferibilmente sui campioni considerati “normali” quando disponibili)
- Training di `sklearn.ensemble.IsolationForest` su feature scalate
- Soglia di anomalia al percentile `100 * (1 - contamination)` degli anomaly score sul training
- Valutazione su `UNSW_top5_eval.csv` con metriche di classificazione

Output salvati:

- `Model/isolation_forest.pkl`: modello IF
- `Model/isoforest_threshold.json`: soglia e metadati
- `Model/scaler_isoforest.pkl`: scaler usato per IF
- `Plots/isoforest_scores.png`: istogramma degli score con soglia

## Requisiti e setup

Prerequisiti:

- Python 3.10+ (consigliato)
- macOS con supporto Apple Silicon opzionale (accelerazione MPS se disponibile)

Installazione dipendenze:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rapido (runner unico)

Esempi principali (AE e IF):

```bash
# 1) Costruzione dataset + training Autoencoder
python run_unsup.py --mode train --type ae --build_dataset --epochs 30 --batch_size 256 --lr 1e-3 --latent_dim 64

# 2) Evaluation Autoencoder (usa i default in Model/ se non specificato)
python run_unsup.py --mode test --type ae --eval_csv Dataset/UNSW_top5_eval.csv

# 3) Training Isolation Forest
python run_unsup.py --mode train --type isoforest --build_dataset --contamination 0.01 --n_estimators 300

# 4) Evaluation Isolation Forest
python run_unsup.py --mode test --type isoforest --eval_csv Dataset/UNSW_top5_eval.csv
```

Opzioni utili:

- `--model_out`, `--threshold_out`: override percorsi di salvataggio
- `--model_in`, `--threshold_in`: percorsi espliciti per la valutazione
- `--device`: per forzare `cpu/cuda/mps`; di default scelto automaticamente

## Note sui label e sulle predizioni

Nel processo di valutazione, le predizioni sono derivate confrontando errore/score con la soglia:

- AE: errori di ricostruzione bassi → inlier (benigno); alti → anomalia
- IF: score alti → più anomali; il confronto con soglia determina la classe

Le metriche vengono riportate nei log insieme alla confusion matrix.

## Cartelle di output

- `Model/`: modelli, soglie, scaler
- `Plots/`: grafici di loss e distribuzioni score
- `Logs/`: file di log con metriche e report

## Sviluppo e riferimenti

- Codice principale dei modelli: `src/network_unsup.py`
- Training/valutazione AE: `src/train_unsup.py`, `src/test_unsup.py`
- Training/valutazione IF: `src/train_isoforest.py`, `src/test_isoforest.py`
- Preparazione dati: `src/data_extraction.py`
- Runner: `run_unsup.py`

Per dettagli metodologici, design e risultati, consultare il documento: Report.pdf.

