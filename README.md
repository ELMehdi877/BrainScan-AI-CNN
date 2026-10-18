# 🧠 BrainScan AI — Classification de Tumeurs Cérébrales par CNN

Ce projet met en œuvre un réseau de neurones convolutif (CNN) développé avec **TensorFlow / Keras** pour détecter et classifier automatiquement les tumeurs cérébrales à partir d'images IRM (*glioma*, *meningioma*, *notumor*, *pituitary*).

---

## 📋 Description du Projet

L'objectif principal est d'assister le diagnostic médical grâce au Deep Learning. Le projet a été conçu et entraîné en environnement local (VS Code / Jupyter Notebook sur CPU) en intégrant une étape d'augmentation de données pour assurer une bonne généralisation du modèle.

### Caractéristiques principales :
- **Prétraitement & Nettoyage :** Redimensionnement des images IRM en $224 \times 224 \times 3$, normalisation $[0, 1]$, et suppression automatique des fichiers corrompus/invalides.
- **Data Augmentation :** Intégration de couches dynamiques (`RandomFlip`, `RandomRotation`, `RandomZoom`) directement dans l'architecture du modèle.
- **Architecture CNN :**
  - Couches convolutives (`Conv2D`) associées à l'activation `ReLU` et au sous-échantillonnage (`MaxPooling2D`).
  - Couches de régularisation (`Dropout` à 50%) pour éviter le surapprentissage.
  - Couche de sortie avec fonction `softmax` à 4 neurones pour la classification multi-classe.
- **Entraînement :** Optimiseur `Adam` ($\text{learning rate} = 0.001$), perte `categorical_crossentropy`, suivi par le callback `ModelCheckpoint` pour la sauvegarde continue du meilleur modèle.
- **Déploiement Web :** Interface utilisateur interactive développée avec **Streamlit** permettant le chargement et l'analyse d'images IRM en direct.

---

## 📁 Structure des Fichiers

```text
BrainScan-AI-CNN/
│
├── Data/
│   ├── Training/          # Images IRM d'entraînement réparties en 4 dossiers de classes
│   └── Testing/           # Images IRM de test réparties en 4 dossiers de classes
│
├── meilleur_modele_brainscan.h5  # Meilleures pondérations sauvegardées par ModelCheckpoint
├── notebook_pretraitement.ipynb  # Notebook Jupyter (préparation, entraînement, évaluation)
├── app.py                         # Application Web Streamlit
├── requirements.txt               # Dépendances du projet
└── README.md                      # Documentation du projet
```

---

## ⚙️ Prérequis et Installation

### 1. Cloner le projet et préparer l'environnement
Assurez-vous d'avoir installé **Python 3.8+** sur votre système.

```bash
# Se placer dans le répertoire du projet
cd BrainScan-AI-CNN

# Installer les dépendances requises
pip install tensorflow numpy matplotlib pillow streamlit
```

---

## 🚀 Instructions d'Exécution

### Étape 1 : Entraînement et Évaluation (Notebook)
1. Ouvrez **VS Code** ou votre environnement Jupyter local.
2. Lancez le fichier `notebook_pretraitement.ipynb`.
3. Exécutez les cellules séquentiellement pour :
   - Charger et nettoyer le jeu de données `Data/`.
   - Compiler et vérifier le modèle (`model.summary()`).
   - Lancer l'entraînement avec Data Augmentation et callbacks.
   - Analyser les courbes d'évolution de la précision (*Accuracy*) et de la perte (*Loss*).

---

### Étape 2 : Lancement de l'Application Web (Streamlit)
Pour démarrer l'interface utilisateur graphique :

1. Ouvrez un terminal dans le dossier racine du projet.
2. Exécutez la commande suivante :
   ```bash
   streamlit run app.py
   ```
3. L'application s'ouvre automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`.
4. Glissez-déposez une image IRM (format `.jpg`, `.jpeg` ou `.png`) et cliquez sur **Analyser l'image** pour obtenir la prédiction et le taux de confiance en pourcentage.

---

## 📊 Résultats d'Entraînement

- **Sans Augmentation de Données :** Convergence rapide atteignant **~97%** d'exactitude (*val_accuracy*).
- **Avec Data Augmentation :** Augmentation de la robustesse face aux variations d'images réelles, avec sauvegarde automatique du meilleur état via `meilleur_modele_brainscan.h5`.