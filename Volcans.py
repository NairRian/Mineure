import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn.datasets import load_wine

"""
values, target = load_wine(return_X_y=True)
target_names = load_wine().target_names
feature_names = load_wine().feature_names
"""
st.title("Les volcans de l'Holocène")

# Chemin du fichier CSV local
csv_path = "C:\Users\camph\Documents\2024-5\S7\database.csv"

# Lecture du fichier CSV avec pandas
try:
    df = pd.read_csv(csv_path)
    # Affichage des données dans l'application
    st.write("Aperçu des données :")
    st.dataframe(df)

    # Optionnel : ajouter des statistiques ou des graphiques
    st.write("Statistiques descriptives :")
    st.write(df.describe())
except FileNotFoundError:
    st.error(f"Le fichier '{csv_path}' est introuvable. Veuillez vérifier le chemin.")
except Exception as e:
    st.error(f"Une erreur s'est produite : {e}")

"""
st.dataframe(df.style.highlight_max(axis=0)) surligne la valeur max de la colonne
plot =df['alcohol']

bot = st.checkbox("Afficher le graphique de la colonne alcohol : ")
if bot :
  st.write('Voici le graphique de la colonne alcohol : ')
  st.line_chart(plot)
"""
