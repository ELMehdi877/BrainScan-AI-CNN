import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="BrainScan AI - Détection de Tumeurs",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 BrainScan AI : Diagnostic IRM")
st.write("Téléversez une image IRM cérébrale pour analyser la présence ou le type de tumeur.")

# Charger le modèle sauvegardé
@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model('scripts/meilleur_modele_brainscan.h5')

model = charger_modele()
nom_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Chargement du fichier image par l'utilisateur
uploaded_file = st.file_uploader("Choisissez une image IRM...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Image IRM téléversée", use_container_width=True)
    
    if st.button("🔍 Analyser l'image"):
        with st.spinner("Analyse du réseau de neurones en cours..."):
            # Redimensionnement identique à l'entraînement (224, 224)
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            
            # S'assurer que les valeurs de pixels sont bien normalisées / 255.0
            img_array = img_array.astype('float32') / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Prédiction
            predictions = model.predict(img_batch)
            indice_pred = np.argmax(predictions[0])
            classe_predite = nom_classes[indice_pred]
            confiance = predictions[0][indice_pred] * 100
            
            # Affichage
            st.markdown("---")
            if classe_predite == 'notumor':
                st.success(f"🟢 **Résultat :** Pas de tumeur détectée ({classe_predite})")
            else:
                st.error(f"🔴 **Résultat :** Tumeur détectée ({classe_predite})")
                
            st.info(f"📊 **Niveau de confiance :** {confiance:.2f}%")