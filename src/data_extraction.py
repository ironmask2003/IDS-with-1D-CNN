import pandas as pd
import numpy as np

INPUT_FILES = [
    "Dataset/UNSW-NB15_1.csv",
    "Dataset/UNSW-NB15_2.csv",
    "Dataset/UNSW-NB15_3.csv",
    "Dataset/UNSW-NB15_4.csv",
]

FEATURES_FILE = "Dataset/NUSW-NB15_features.csv"
LABEL_COLUMN = "Label"
ATTACK_COLUMN = "attack_cat"

OUTPUT_TRAIN = "Dataset/UNSW_top5_train.csv"
OUTPUT_EVAL = "Dataset/UNSW_top5_eval.csv"

TOP_ATTACKS_TRAIN_RATIO = 0.8   # 80% dei top5 attacchi in train


def load_with_header(csv_path):
    cols = pd.read_csv(FEATURES_FILE, encoding="latin1")["Name"].tolist()
    return pd.read_csv(csv_path, header=None, names=cols, encoding="latin1")

def build_datasets():
    # -------------------------------------------------------------------------
    # 1. MERGE COMPLETO
    # -------------------------------------------------------------------------
    df_list = [load_with_header(p) for p in INPUT_FILES]
    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    print(f"Dataset totale: {len(df)} righe")

    # -------------------------------------------------------------------------
    # 2. IDENTIFICA I TOP 5 ATTACCHI
    # -------------------------------------------------------------------------
    attack_counts = df[ATTACK_COLUMN].value_counts()
    top5 = attack_counts.head(5).index.tolist()

    print("\nTop 5 attacchi più frequenti:")
    for att in top5:
        print(f"  {att}: {attack_counts[att]} record")


    # -------------------------------------------------------------------------
    # 3. SEPARA I DATI PER ATTACCHI TOP5 E ALTRI
    # -------------------------------------------------------------------------
    df_top5 = df[df[ATTACK_COLUMN].isin(top5)]
    df_other = df[~df[ATTACK_COLUMN].isin(top5)]
    df_other = df_other[df_other[LABEL_COLUMN] == 1]  # solo attacchi

    print(f"\nTop5 attacchi: {len(df_top5)} righe")
    print(f"Altri attacchi: {len(df_other)} righe")

    # -------------------------------------------------------------------------
    # 4. SALVA I DATI DEI TOP5
    # -------------------------------------------------------------------------
    df_top5_pos = df_top5[df_top5[LABEL_COLUMN] == 1]

    n_train_top5 = int(len(df_top5_pos) * TOP_ATTACKS_TRAIN_RATIO)

    # training
    train_pos_top5 = df_top5_pos.sample(n_train_top5, random_state=42)
    eval_pos_top5 = df_top5_pos.drop(train_pos_top5.index)
    df_label0 = df[df[LABEL_COLUMN] == 0]
    print(f"Non-attacchi totali: {len(df_label0)} righe")
    eval_pos_other = df_other[df_other[LABEL_COLUMN] == 1]
    eval_neg = df_label0.sample(len(eval_pos_other), random_state=42)
    print(f"Training top5 attacchi: {len(train_pos_top5)} righe")
    print(f"Eval records: esperienze positive = {len(eval_neg)} + esperienze non top 5  = {len(eval_pos_other)} righe")
    # evaluation
    #eval_df = pd.concat([eval_pos_top5, eval_neg, eval_pos_other], axis=0).sample(frac=1, random_state=42)
    eval_df = pd.concat([eval_neg, eval_pos_other], axis=0).sample(frac=1, random_state=42)
    # taglio delle label solo sul training (unsupervised). L'eval mantiene tutte le feature + Label.
    train_pos_top5 = train_pos_top5.drop(columns=[LABEL_COLUMN, ATTACK_COLUMN])
    train_pos_top5.to_csv(OUTPUT_TRAIN, index=False)
    # Manteniamo tutte le colonne in eval (feature + Label + attack_cat) per permettere il caricamento con UNSWDataset
    eval_df.to_csv(OUTPUT_EVAL, index=False)

