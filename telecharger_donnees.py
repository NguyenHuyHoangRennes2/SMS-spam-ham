"""Télécharge les deux jeux de données du projet depuis Hugging Face et les enregistre dans data/.

À lancer une seule fois (les fichiers CSV produits sont fournis avec le projet) :
    pip install datasets huggingface_hub pandas
    python telecharger_donnees.py

Fichiers produits (colonnes text, label) :
    data/sms_brut.csv    : Kinoux/french-spam-ham-detection-free-2k (1 000 messages en français)
    data/sms_brut_2.csv  : SMS Spam Collection traduite en français (5 572 messages)
"""
import pandas as pd
from datasets import load_dataset

# 1. Kinoux/french-spam-ham-detection-free-2k
df = load_dataset("Kinoux/french-spam-ham-detection-free-2k")["train"].to_pandas()
df[["text", "label"]].to_csv("data/sms_brut.csv", index=False, encoding="utf-8")
print("data/sms_brut.csv", df.shape)

# 2. dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset : on garde la traduction française
df = pd.read_csv("hf://datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset/data-augmented.csv")
df = pd.DataFrame({
    "text": df["text_fr"],
    "label": df["labels"],
})
df.to_csv("data/sms_brut_2.csv", index=False, encoding="utf-8")
print("data/sms_brut_2.csv", df.shape)
