import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Projet ML - Vin",
    page_icon=":cat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd 
import accueil as page0
import visualisations as page1
import modelisation as page2
import upload as page3

# Sélection des pages de navigation
if "page" not in st.session_state:
    st.session_state.page = "Accueil"

def navigate_to(page):
    st.session_state.page = page

# Création des boutons de navigation entre les pages
st.sidebar.button("Accueil", on_click=navigate_to, args=("Accueil",), icon="🏠")
st.sidebar.button("Visualisation des données", on_click=navigate_to, args=("Visualisation des données",), icon="📊")
st.sidebar.button("Machine Learning - Evaluation", on_click=navigate_to, args=("Machine Learning - Evaluation",), icon="📈")
st.sidebar.button("A vous de jouer !", on_click=navigate_to, args=("A vous de jouer !",), icon="📄")

# Afficher la bonne page
if st.session_state.page == "Accueil":
    page0.show()
elif st.session_state.page == "Visualisation des données":
    page1.show()
elif st.session_state.page == "Machine Learning - Evaluation":
    page2.show()
elif st.session_state.page == "A vous de jouer !":
    page3.show()

