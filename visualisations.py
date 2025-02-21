import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

df_vin = pd.read_csv(r"D:\Diginamic\2. Cours\34. Conception Et Developpement D'une IA\data_vin.csv")

def show():
    st.title("Visualisation des données")
    st.text("Cette page rassemble les différentes statistiques et visualisations sous forme de graphique des données du DataFrame exemple. Choississez les colonnes à analyser et naviguez entre les différents onglets !")
      
    selection_colonnes = st.multiselect("**Sélectionner les colonnes à analyser**", df_vin.columns.tolist(), default=df_vin.columns.tolist())
    df_vin_selection = df_vin[selection_colonnes]
    
    numerique_cols = df_vin_selection.select_dtypes(include='number').columns   
    
    tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Statistiques générales", "Histogrammes", "Pairplot", "Corrélations"])

    with tabs_1:
        # Créer des colonnes
        col1, col2 = st.columns([1, 2])  # 1: largeur de la colonne image, 2: largeur de la colonne texte

        # Afficher la fréquence dans la première colonne
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            count = df_vin["target"].value_counts()
            bars = ax.bar(count.index, count.values, color=list(mcolors.TABLEAU_COLORS.values())[:len(count)])
            # Ajouter les valeurs sur les barres
            for bar in bars.patches:
                ax.annotate(format(bar.get_height()),
                            (bar.get_x() + bar.get_width() / 2,
                            bar.get_height() /2), ha='center', va='center',
                            size=20, color='white', weight='bold')
            ax.set_title("Répartition des vins en fonction de leur type", size=18)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            st.pyplot(fig)
            plt.close(fig)
            
        with col2:
            # Nombre de lignes et colonnes du DataFrame
            st.markdown(f"*Nombre de lignes du DataFrame : **{len(df_vin)}***")
            st.markdown(f"*Nombre de colonnes du DataFrame : **{len(df_vin.columns)}***")
    
        # Affichage du DataFrame
        st.subheader("**Dataframe vin.csv**")
        st.dataframe(df_vin)
        
        # Créer des colonnes
        col1, col2 = st.columns([1, 2])  # 1: largeur de la colonne image, 2: largeur de la colonne texte

        # Afficher l'image dans la première colonne
        with col1:
            # Vérification de la présence de valeurs nulles dans le DataFrame
            st.write("*Nombre de valeurs nulles dans le DataFrame :*", df_vin.isnull().sum())

        # Afficher le texte dans la deuxième colonne
        with col2:
            # Analyse descriptive
            st.write("*Statistiques du DataFrame :*", df_vin.describe().T)

    with tabs_2:
        st.subheader("Histogramme")
        st.write("*Un histogramme est une représentation graphique de la distribution de données numériques. Il divise les données en intervalles (appelés classes ou bins) et affiche le nombre de données qui se trouvent dans chaque intervalle sous forme de barres.*")
        
        # Mise en forme de l'affichage des histogrammes
        num_cols = len(numerique_cols)
        num_rows = 5
        cols_par_row = (num_rows + num_cols - 1) // num_rows
        
        # Création des lignes de colonnes
        rows = []
        for i in range(num_rows):
            rows.append(st.columns(cols_par_row))
    
        # Affichage des histogrammes
        for i, col in enumerate(numerique_cols):
            row_index = i // cols_par_row  
            col_index = i % cols_par_row  
            fig, ax = plt.subplots(figsize=(6, 4))  
            df_vin_selection[col].plot.hist(bins=50, ax=ax)
            ax.set_title(f"Histogramme de {col}")
            rows[row_index][col_index].pyplot(fig) 
        pass

    with tabs_3:
        # Affichage du Pairplot
        st.subheader("Pairplot")
        st.write("*Un pairplot (ou graphique de paires) est une technique de visualisation de données qui permet d'explorer les relations entre plusieurs variables numériques dans un jeu de données. Il s'agit d'une matrice de graphiques, où chaque graphique représente la relation entre deux variables.*")
        st.pyplot(sns.pairplot(df_vin_selection.select_dtypes(include='number'), height=3))
        plt.close(fig)
        pass

    with tabs_4:
        # Affichage de la Heatmap
        st.subheader("Corrélations (Heatmap)")
        st.write("*Une heatmap (ou carte thermique) est une représentation graphique de données où les valeurs individuelles d'une matrice sont représentées par des couleurs.  C'est un excellent moyen de visualiser des données tabulaires, en particulier lorsque vous avez beaucoup de données à comparer.*")
        fig, ax = plt.subplots()
        sns.heatmap(df_vin_selection.select_dtypes(include='number').corr(), annot=True, ax=ax, annot_kws={"size": 6}, cbar_kws={"shrink": .8})
        ax.tick_params(labelsize=6)
        st.pyplot(fig)
        plt.close(fig)
        pass
