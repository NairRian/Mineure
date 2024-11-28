import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import math

st.title("Les volcans de l'Holocène")

st.write("Exposer nos idées : ")

# Chemin du fichier CSV local
csv_path = "./database.csv"

# Lecture du fichier CSV avec pandas
df = pd.read_csv(csv_path)
# Affichage des données dans l'application
st.write("Aperçu des données :")
st.dataframe(df)

# Optionnel : ajouter des statistiques ou des graphiques
st.write("Statistiques descriptives :")
st.write(df.describe())

# afficher la version traitée du document
st.write("Nous avons effectué les modifications suivantes : ")
# dff = modif de df
# st.dataframe(dff)

# ANALYSE SELON TYPE DE ROCHES, D'ERUPTION ET DE TECTONIQUE
# réduire l'excel aux colonnes qui nous intéressent
data_infos = data[['Type', 'Dominant Rock Type', 'Tectonic Setting']]
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
type_stats




#st.dataframe(df.style.highlight_max(axis=0)) surligne la valeur max de la colonne
#plot =df['alcohol']

#bot = st.checkbox("Afficher le graphique de la colonne alcohol : ")
#if bot :
#  st.write('Voici le graphique de la colonne alcohol : ')
#  st.line_chart(plot)
