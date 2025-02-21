import streamlit as st
from PIL import Image

def show():
    st.title("De la Donnée au Modèle : Voyage au cœur de la qualité du vin")
    st.write("Embarquez pour un voyage interactif au cœur de la qualité du vin !")
    st.write("Découvrez cette application Streamlit au travers d'un jeu de données qui vous conduira sur les étapes clés d'un projet de Machine Learning, de la manipulation des données à la construction et à l'évaluation de modèles prédictifs.")
    
    image = Image.open("Vins.jpg")
    new_size = (500, 350)  # Largeur, Hauteur
    resized_image = image.resize(new_size)
    
    # Créer des colonnes
    col1, col2 = st.columns([1, 2])  # 1: largeur de la colonne image, 2: largeur de la colonne texte

    # Afficher l'image dans la première colonne
    with col1:
        st.image(resized_image)

    # Afficher le texte dans la deuxième colonne
    with col2:
        st.write("Naviguez à travers les différentes sections pour :")
        st.markdown("- **Explorer et visualiser les données :** A l'aide du jeu de données fourni, visualiser et analyser les informations.")
        st.markdown("- **Mettre en œuvre un pipeline de Machine Learning :** Choisissez un algorithme, entraînez un modèle et faites des prédictions.")
        st.markdown("- **Évaluer les performances du modèle :** Analysez les métriques et les visualisations pour comprendre la qualité des prédictions.")
        st.markdown("- **A vous de jouer ! :** Télécharger votre propre jeu de données et explorer les étapes d'un projet de Machine Learning.")