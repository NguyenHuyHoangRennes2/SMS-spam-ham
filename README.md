# Détection de SMS frauduleux (spam / smishing) en français

Projet de fouille de données textuelles — Master 2 MAS (SDD), Université Rennes 2.

Classification automatique de SMS en **ham** (légitime) ou **spam** (publicité abusive, hameçonnage) à partir de deux corpus combinés, avec spaCy et scikit-learn.

## Résultats

| | Valeur |
|---|---|
| Corpus | 6 105 messages (18,7 % de spams), issus de deux jeux de données |
| Modèle retenu | SVM linéaire sur TF-IDF |
| F1 (classe spam, jeu de test) | 0,964 |
| AUC-ROC | 0,995 |

Une expérience croisée montre qu'un modèle entraîné sur un seul corpus généralise mal à l'autre (F1 de 0,64 et 0,53), alors que le modèle entraîné sur les deux corpus réussit sur chacun (0,99 et 0,94).

## Démarche

1. **Données** : `Kinoux/french-spam-ham-detection-free-2k` (1 000 messages) et la *SMS Spam Collection* traduite en français (`dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`, 5 572 messages).
2. **Nettoyage** : suppression des doublons, masquage des URL, montants, pourcentages et numéros par des balises (`<URL>`, `<PRIX>`…), lemmatisation et filtrage des stop-words avec spaCy (`fr_core_news_md`).
3. **Modélisation** : TF-IDF, comparaison de la régression logistique, du k-NN, du SVM linéaire et de la Random Forest par validation croisée à 5 plis ; comparaison de `class_weight` et de SMOTE pour le déséquilibre des classes.
4. **Évaluation** : précision, rappel, F1, AUC-ROC, matrice de confusion, analyse des erreurs et des coefficients du modèle.

Le détail des choix et des résultats est dans [`RESUME_PROJET.pdf`](RESUME_PROJET.pdf).

## Contenu

| Fichier | Rôle |
|---|---|
| `projet_sms_spam.ipynb` | Notebook complet : code commenté, résultats et analyses |
| `telecharger_donnees.py` | Téléchargement des deux jeux de données depuis Hugging Face |
| `data/sms_brut.csv`, `data/sms_brut_2.csv` | Données initiales |
| `data/sms_nettoye.csv` | Messages nettoyés |
| `data/predictions_test.csv` | Prédictions du modèle final sur le jeu de test |
| `RESUME_PROJET.pdf` | Rapport de synthèse |

## Exécution

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
jupyter notebook projet_sms_spam.ipynb
```

Le notebook est enregistré avec ses résultats et peut être lu sans rien installer.
