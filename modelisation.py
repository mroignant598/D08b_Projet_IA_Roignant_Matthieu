import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, pipeline, compose, metrics, tree, neural_network, linear_model, ensemble
from math import sqrt

# Chargement des données
df_vin = pd.read_csv("data_vin.csv")

# Remplacement des catégories par des valeurs numériques
df_vin = df_vin.replace({"Vin amer": 0, "Vin equilibre": 1, "Vin sucre": 2})

def show():
    st.title("Machine Learning - Évaluation des modèles")
    st.write("Analysez la performance du modèle au travers des différents algorithmes, en utilisant des métriques clés et des visualisations pour juger de leur efficacité.")
    
    # Sélection des colonnes à analyser
    selection_colonnes = st.multiselect("**Sélectionner les colonnes à analyser**", df_vin.columns.tolist(), default=df_vin.columns.tolist())
    df_vin_selection = df_vin[selection_colonnes]

    # Séparation des données en features (X) et target (y)
    X = df_vin_selection.drop(columns=['target'])
    y = df_vin_selection['target']

    # Normalisation des données
    scaler = preprocessing.StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Découpage en jeu d'entraînement et test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.15, random_state=42)

    # Création des onglets dans l'interface Streamlit
    tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Modèle Linéaire", "Arbre de Décision", "Réseau de neurones", "RandomForest"])

    # Onglet 1 - Modèle linéaire
    with tabs_1:
        tabs_10, tabs_11, tabs_12 = st.tabs(["Régression Linéaire", "Validation Croisée", "Régression Polynomiale"])

        with tabs_10:
            # Création du modèle de régression linéaire
            st.subheader("Régression Linéaire")
            st.write("*La régression linéaire est une méthode statistique utilisée pour modéliser la relation entre une variable dépendante (cible) et une ou plusieurs variables indépendantes (prédicteurs). L'objectif est de trouver une équation linéaire qui décrit au mieux la relation entre ces variables, permettant ainsi de prédire la valeur de la variable dépendante à partir des valeurs des variables indépendantes.*")
            st.write("*Plus la vleur du RMSE est faible, meilleures sont les performances du modèle. Plus le R² est élevé, meilleur est l'ajustement du modèle aux données.*")
            regression = linear_model.LinearRegression()
            regression.fit(X_train, y_train)
            test_predictions = regression.predict(X_test)

            # Affichage des résultats
            col1, col2 = st.columns([1, 2])

            with col1:
                # Graphique : Comparaison entre les prédictions et les valeurs réelles
                fig, ax = plt.subplots()
                ax.scatter(y_test, test_predictions, color='black', label="Prédictions")
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'red', lw=1, label="Idéal")
                ax.set_title("Prédictions vs Valeurs Réelles")
                ax.set_xlabel("Valeurs Observées")
                ax.set_ylabel("Prédictions")
                ax.legend()
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                # Affichage des performances du modèle
                st.write(f"RMSE (Erreur Quadratique Moyenne) = {round(sqrt(metrics.mean_squared_error(y_test, test_predictions)),2)}")
                st.write(f"R² Score = {round(metrics.r2_score(y_test, test_predictions),2)}")

        with tabs_11:
            st.subheader("Validation croisée avec KFold sur le modèle de régression linéaire")

            def cross_validation_model(k=10):
                """Effectue une validation croisée avec KFold sur le modèle de régression linéaire"""
                kf = model_selection.KFold(n_splits=k, shuffle=True, random_state=42)
                model = linear_model.LinearRegression()

                rmse_scores = []
                r2_scores = []

                for train_index, test_index in kf.split(X_scaled):
                    X_train_k, X_test_k = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train_k, y_train_k)
                    predictions = model.predict(X_test_k)

                    rmse = sqrt(metrics.mean_squared_error(y_test_k, predictions))
                    r2 = metrics.r2_score(y_test_k, predictions)

                    rmse_scores.append(rmse)
                    r2_scores.append(r2)

                # Affichage des résultats de la validation croisée
                st.write(f"**Validation croisée ({k}-Fold) :**")
                st.write(f"- Moyenne RMSE (Erreur Quadratique Moyenne) : {round(np.mean(rmse_scores), 2)}")
                st.write(f"- Moyenne R² : {round(np.mean(r2_scores), 2)}")

                # Affichage sous forme de dataframe pour plus de clarté
                results_df = pd.DataFrame({"Fold": list(range(1, k+1)), "RMSE": rmse_scores, "R²": r2_scores})
                st.dataframe(results_df)

            cross_validation_model(k=20)
            
        with tabs_12:
            st.subheader("Régression Polynomiale")
            st.write("*La régression polynomiale est une forme de régression linéaire dans laquelle la relation entre la variable indépendante (x) et la variable dépendante (y) est modélisée 1  comme un polynôme de degré n. Elle est utilisée lorsque la relation entre les variables n'est pas linéaire.*")
            st.write("*Plus la vleur du RMSE est faible, meilleures sont les performances du modèle. Plus le R² est élevé, meilleur est l'ajustement du modèle aux données.*")
            # Création du modèle avec pipeline pour ajouter les features polynomiales
            model = pipeline.Pipeline([
                ("scaler", preprocessing.StandardScaler()),  # Normalisation des données
                ("polynomial_features", preprocessing.PolynomialFeatures(degree=2)),  # Génération des features polynomiales
                ("linear_regression", linear_model.LinearRegression())  # Régression linéaire
            ])
            # Entraînement du modèle
            model.fit(X_train, y_train)

            # Prédictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            # Affichage des résultats
            col1, col2 = st.columns([1, 2])

            with col1:
                # Graphique : Prédictions vs Réel
                fig, ax = plt.subplots()
                ax.scatter(y_test, test_predictions, color='black', alpha=0.7, label="Prédictions")
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'red', lw=1, label="Idéal")
                ax.set_title("Prédictions vs Valeurs Réelles")
                ax.set_xlabel("Valeurs Observées")
                ax.set_ylabel("Prédictions")
                ax.legend()
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                # Calcul et affichage des scores de performance du modèle (RMSE et R²)
                rmse = sqrt(metrics.mean_squared_error(y_test, test_predictions))
                r2 = metrics.r2_score(y_test, test_predictions)
                st.write(f"**RMSE (Erreur Quadratique Moyenne) :** {round(rmse, 2)}")
                st.write(f"**R² Score :** {round(r2, 2)}")
                
    with tabs_2:
        st.subheader("Arbre de Décision")
        st.write("*Un arbre de décision est un algorithme d'apprentissage automatique supervisé utilisé pour la classification et la régression. Il fonctionne en divisant les données en sous-ensembles de plus en plus petits en fonction de caractéristiques spécifiques. Voici les concepts clés concernant les arbres de décision :*")
        st.write("   - *Nœud racine : Le nœud de départ de l'arbre, qui représente l'ensemble de données initial.*")
        st.write("   - *Nœuds internes : Les nœuds qui représentent les caractéristiques utilisées pour diviser les données.*")
        st.write("   - *Branches : Les lignes qui relient les nœuds et représentent les décisions prises en fonction des caractéristiques.*")
        st.write("   - *Nœuds feuilles : Les nœuds finaux de l'arbre, qui représentent les prédictions ou les classifications.*")
        st.write("*'Accuracy on train set' = 'Précision sur l'ensemble d'entraînement' - Une précision élevée sur l'ensemble d'entraînement peut indiquer que le modèle a bien appris les motifs présents dans les données d'entraînement.*")

        # Création d'un pipeline avec mise à l'échelle des données et l'arbre de décision
        pipe = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),  
        ('decision_tree', tree.DecisionTreeClassifier(random_state=42)) 
        ])
        
        # Entraînement du modèle
        pipe.fit(X_train, y_train)
        
        # Affichage des précisions sur le jeu d'entraînement et de test
        st.write("Accuracy on train set =", pipe.score(X_train,y_train))
        st.write("Accuracy on test set =", pipe.score(X_test,y_test))
        
        # Affichage de l'arbre de décision
        fig, ax = plt.subplots(figsize=(20, 10))
        tree.plot_tree(
            pipe.named_steps['decision_tree'],  
            feature_names=X_train.columns,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax  
            )
        st.pyplot(fig)
        plt.close(fig)
            
        # Affichage des résultats
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Matrice de confusion")
            # Affichage de la matrice de confusion
            y_pred = pipe.predict(X_test)
            fig, ax = plt.subplots(figsize=(5, 5))
            cm_display = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # **Affichage des métriques sous forme de DataFrame**
            def classification_report_to_df(report):
                report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
                return pd.DataFrame(report_dict).transpose()

            st.subheader("Métriques de classification")
            report_df = classification_report_to_df(metrics.classification_report(y_test, y_pred, output_dict=True))
            st.dataframe(report_df.style.format(precision=2))

    with tabs_3:
        st.subheader("Réseau de Neurone")
        st.write("*Un réseau de neurones, ou réseau neuronal, est un modèle d'apprentissage automatique inspiré du fonctionnement du cerveau humain. Il est composé de couches de neurones artificiels interconnectés, capables d'apprendre des motifs complexes à partir de données.*")
        st.write("*'Accuracy on train set' = 'Précision sur l'ensemble d'entraînement' - Une précision élevée sur l'ensemble d'entraînement peut indiquer que le modèle a bien appris les motifs présents dans les données d'entraînement.*")
        # Création du pipeline avec mise à l'échelle des données et réseau de neurones (MLP)
        pipe = pipeline.Pipeline([
        ('std_scaler', preprocessing.StandardScaler()),
        ('neural_network', neural_network.MLPClassifier(random_state=42, max_iter=500))]
        )

        pipe.fit(X_train, y_train)
        
        # Affichage des résultats
        col1, col2, col3 = st.columns([1, 2, 3])

        with col1:
            st.write("Accuracy on train set =", pipe.score(X_train,y_train))
            st.write("Accuracy on test set =", pipe.score(X_test,y_test))

        with col2:
            st.subheader("Matrice de confusion")
            # Affichage de la matrice de confusion
            y_pred = pipe.predict(X_test)
            fig, ax = plt.subplots(figsize=(5, 5))
            cm_display = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
            st.pyplot(fig)
            plt.close(fig)


        with col3:
            # **Affichage des métriques sous forme de DataFrame**
            def classification_report_to_df(report):
                report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
                return pd.DataFrame(report_dict).transpose()

            st.subheader("Métriques de classification")
            report_df = classification_report_to_df(metrics.classification_report(y_test, y_pred, output_dict=True))
            st.dataframe(report_df.style.format(precision=2))
            
    with tabs_4:
        st.subheader("Random Forest")
        st.write("*Un Random Forest, ou forêt aléatoire, est un algorithme d'apprentissage automatique d'ensemble très populaire et puissant. Il est utilisé pour la classification et la régression.*")
        # Affichage des résultats
        col1, col2 = st.columns([1, 2])

        with col1:
            # Création d'un modèle Random Forest pour l'évaluation
            random = ensemble.RandomForestClassifier()
            random.fit(X_train, y_train)
            y_pred = random.predict(X_test)
            
            # Calcul de la précision du modèle
            acc = metrics.accuracy_score(y_test, y_pred)
            
            fig, ax = plt.subplots()
            sns.histplot(y_pred, kde=True, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Affichage de la précision du modèle
            st.write(f"Précision du modèle : {acc:.2f}")