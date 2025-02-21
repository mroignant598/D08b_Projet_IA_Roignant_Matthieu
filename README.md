Projet Machine Learning 

Equipe : D08b - Matthieu Roignant

Processus de développement de l'application :
- Mise en forme de streamlit, répartition sur 4 pages
- Page Accueil : Présentation de l'application
- Page Visualisation des données : regroupe des statistiques générales sur le jeu de données exemple et plusieurs types de graphiques
- Page Machine Learning - Evaluation : A partir du jeu de données exemple, exploration des différents algorithmes de Machine Learning
- Page A vous de jouer ! : Permet d'insérer son propre jeu de données et d'explorer les visuels et les algorithmes de Machine Learning pour évaluer son jeu de données

Méthode de travail : 
- Mise en place du site
- Développement des fonctionnalités de bases présentes dans l'onglet Visualisation des données sur le jeu de données exemple
- Développement des fonctionnalités de Machine Learning sur le jeu de données exemple
- Mise en place de l'onglet A vous de jouer ! pour insérer un jeu de données, le nettoyer et l'utiliser dans les différentes fonctionnalités de Machine Learning

Mode d'emploi de l'application :
- Placer les fichiers (app.py, accueil.py, visualisations.py, modelisation.py, upload.py, nettoyage_vin.py, Vins.jpg, vin.csv) dans le même dossier 
- Ouvrir nettoyage_vin.py dans VSC et l'éxécuter
- activer l'environnement virtuel = env-MLVIN\Scripts\activate 
- lancer streamlit = python -m streamlit run app.py
- Vous pouvez naviguer dans les différentes pages de l'application pour découvrir le jeu de données exemple 
- Allez sur la page 'A vous de jouer !', insérer votre jeu de données et laissez vous guider en suivant les différentes étapes, puis naviguez dans les différents onglets (si il a des valeurs NaN, remplacez les par 0) (Si vous avez des valeurs string, activez le LabelEncoder() pour les remplacer par des valeurs numériques)
