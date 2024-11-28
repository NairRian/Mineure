import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# DATA
# Chemin du fichier CSV local
csv_path = "./database.csv"

# Lecture du fichier CSV avec pandas
df = pd.read_csv(csv_path)

###################################################################
# transformation du fichier

#Parcourir la colonne 'Type' dans le dataframe data, pour chaque valeur texte finissant par '(s)' ou '(es)', supprimer '(s)' ou '(es)'
# attention si l'information n'est pas renseignée, erreur : ajouter condition vérifiant que c un string
suffixes = ['(s)', '(es)', '?']  # Liste des suffixes à supprimer

for i in range(len(df)):
    if isinstance(df['Type'].iloc[i], str):
        for suffix in suffixes:
            if df['Type'].iloc[i].endswith(suffix):
                df['Type'].iloc[i] = df['Type'].iloc[i].replace(suffix, '')

# changer les dates

def convert_date(date_str):
    if isinstance(date_str, str):
        if 'BCE' in date_str:
            try:
                year = int(date_str.split(' BCE')[0])
                return -year
            except ValueError:
                return np.nan
        elif 'CE' in date_str:
            try:
                year = int(date_str.split(' CE')[0])
                return year
            except ValueError:
                return np.nan
        else:
            try:
                year = int(date_str)
                return year
            except ValueError:
                return np.nan
    return np.nan
for i in range(len(df)):
    if isinstance(df['Last Known Eruption'].iloc[i], str):
            df['Last Known Eruption'].iloc[i] = convert_date(df['Last Known Eruption'].iloc[i])

###################################################################

# configurer la page
apptitle = "Volcans de l'Holocène - GP5"
st.set_page_config(page_title=apptitle, page_icon="🌋")

st.title("Les volcans de l'Holocène")

st.markdown("""
 * Utiliser le menu pour choisir ce que vous souhaitez afficher
 * Ci-dessous parait vos choix
""")

###################################################################

# configuration de la sidebar
st.sidebar.title("Choix des informations que vous souhaitez afficher : ")

###################################################################
st.sidebar.subheader("Informations volcanologiques pour un pays")

# PARTIE ALEXANDRE
st.subheader(f"Informations volcanologiques  : {pays}")
#Choix du nom du pays
liste_pays = list(set(df['Country'].values))
pays = st.sidebar.selectbox("Quel pays ?", liste_pays)
alex = st.sidebar.checkbox("Afficher infos du pays")

if alex :
    data_pays = df[df['Country'] == pays][['Country', 'Type', 'Last Known Eruption', 'Name']]
    #Condition pour si le pays choisi n'est pas dans la database
    if pays in df['Country'].values:
        # Trouver la dernière date d'éruption
        derniere_eruption = max(data_pays['Last Known Eruption'])
        
        # Récupérer le ou les noms des volcans associés à cette éruption
        volcans_derniere_eruption = data_pays[data_pays['Last Known Eruption'] == derniere_eruption]['Name']
        
        # Compter les types d'éruption
        type_eruption = data_pays['Type'].value_counts()
        
        # Affichage des résultats
        st.write(f"La dernière éruption connue est datée de {derniere_eruption}.")
        st.write(f"Le(s) volcan(s) associé(s) à cette éruption : {', '.join(volcans_derniere_eruption)}")
        st.write("Type d'éruptions dans ce pays :\n", type_eruption)
    else:
        st.write(f"Le pays {pays} n'est pas le jeu de donné.")

###################################################################
st.subheader("Lien entre les types de roche, tectonique et éruption")

# PARTIE 1 RAPH
st.sidebar.subheader("Lien entre les types de roche, de tectonique et d'éruption")

# ANALYSE SELON TYPE DE ROCHES, D'ERUPTION ET DE TECTONIQUE
# réduire l'excel aux colonnes qui nous intéressent
data_infos = df[['Type', 'Dominant Rock Type', 'Tectonic Setting']]
# créer une liste des éléments dans chaque colonne sans doublons
liste_type = list(set(data_infos['Type'].values))
liste_rock = list(set(data_infos['Dominant Rock Type'].values))
liste_tecto = list(set(data_infos['Tectonic Setting'].values))
# grouper la data par 'Dominant Rock Type' and 'Tectonic Setting', et compter les itérations de chaque 'Type', si un 'Type' n'est pas représenté, indiquer 0
type_stats = (
    data_infos.groupby(['Dominant Rock Type', 'Tectonic Setting'])['Type']
    .value_counts() # compter les itérations
    .rename_axis(['Dominant Rock Type', 'Tectonic Setting', 'Type'])  # Ajouter les intitulés
    .unstack(fill_value=0)  # si valeur nulle
)


# afficher le graphique pour un binôme
# demander le binôme
rock_type = st.sidebar.selectbox("Quel type de roche ?", liste_rock)
tectonic_setting = st.sidebar.selectbox("Quel cas tectonique ?", liste_tecto)
raph = st.sidebar.checkbox("Afficher le graphique")

if raph :
    st.write(f"Vous avez choisi : {rock_type} et {tectonic_setting}")
    if (rock_type, tectonic_setting) in type_stats.index:
        type_counts = type_stats.loc[(rock_type, tectonic_setting)] # loc permet de créer un tableau qui répertorie les types de roche et leur itération
        type_counts = type_counts[type_counts != 0] # suppression des valeurs = 0
        tot = sum(type_counts)
        for i in type_counts.index:  # Parcourir les index
            type_counts[i] = type_counts[i] / tot * 100
        
        st.write("Voici le graphique obtenu :")
    
        # Création du graphique Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        type_counts.plot(kind='bar', ax=ax)
        ax.set_title(f"Distribution des types d'éruption pour la combinaison : {rock_type} & {tectonic_setting}")
        ax.set_xlabel("Type d'éruption")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel("Pourcentage de distribution")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        plt.tight_layout()
        
        # Affichage dans Streamlit
        st.pyplot(fig)
    else:
        st.write(f"La combinaison indiquée n'existe pas : {rock_type} & {tectonic_setting}")


###################################################################

# PARTIE 2 RAPH

# prompt: pour un type d'éruption 'Type', représenter sur un graphique à double entrées (en ordonnée les valeurs de 'Dominant Rock Type', en abscisse les valeurs de 'Tectonic Setting') la probabilité d'occurence
# en considérant que la probabilité d'occurence est égale à la probabilité que ce binôme 'Dominant Rock Type' et 'Tectonic Setting' donne ce type d'éruption par rapport à tous les types d'éruption qu'il peut former

# demander le type d'éruption
eruption_type = st.sidebar.selectbox("Quel type d'éruption ?", liste_type)
raphbis = st.sidebar.checkbox("Afficher la heatmap")

if raphbis :
    st.write(f"Vous avez choisi : {eruption_type}")
    
    if eruption_type in type_stats.columns:
        # Calculer les probabilités pour le type d'éruption donné
        probabilities = type_stats[eruption_type] / type_stats.sum(axis=1) * 100
    
        st.write("Voici la heatmap obtenue :")
        # Création de la heatmap
        probabilities_swap = probabilities.swaplevel()  # Échanger abscisses et ordonnées
        heatmap_data = probabilities_swap.unstack()  # Transformer en DataFrame pour heatmap
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="viridis",
            fmt=".1f",
            ax=ax
        )
        ax.set_title(f"Probabilité d'occurrence de l'éruption : '{eruption_type}'")
        ax.set_xlabel("Situation tectonique")
        ax.set_ylabel("Type de roche")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Affichage dans Streamlit
        st.pyplot(fig)
    else:
        st.write(f"Le type d'éruption '{eruption_type}' n'existe pas dans les données.")

