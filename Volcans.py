import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
type_stats

# afficher le graphique pour un binôme
# demander le binôme
#rock_type = input("Quel type de roche ? attention à respecter l'écriture correcte : ")
#tectonic_setting = input("Quel cas tectonique ? attention à respecter l'écriture correcte : ")

rock_type = st.selectbox("Quel type de roche ?", liste_rock)
tectonic_setting = st.selectbox("Quel cas tectonique ?", liste_tecto)
st.write(f"Vous avez choisi : {rock_type} et {tectonic_setting}")

if (rock_type, tectonic_setting) in type_stats.index:
  type_counts = type_stats.loc[(rock_type, tectonic_setting)] # loc permet de créer un tableau qui répertorie les types de roche et leur itération
  type_counts = type_counts[type_counts != 0] # suppression des valeurs = 0
  tot = sum(type_counts)
  for i in type_counts.index:  # Parcourir les index
    type_counts[i] = type_counts[i] / tot * 100

  # Create a bar plot
  plt.figure(figsize=(10, 6))
  type_counts.plot(kind='bar')
  plt.title(f"Distribution des types d'éruption pour la combinaison : {rock_type} & {tectonic_setting}")
  plt.xlabel("Type d'éruption")
  plt.xticks(rotation=45, ha='right')
  plt.ylabel('Pourcentrage de distribution')
  plt.ylim(0, 100)
  plt.yticks(range(0,101,10))
  plt.tight_layout()
  plt.show()
else:
  print(f"La combinaison indiquée n'existe pas : {rock_type} & {tectonic_setting}")


#st.dataframe(df.style.highlight_max(axis=0)) surligne la valeur max de la colonne
#plot =df['alcohol']

#bot = st.checkbox("Afficher le graphique de la colonne alcohol : ")
#if bot :
#  st.write('Voici le graphique de la colonne alcohol : ')
#  st.line_chart(plot)
