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
st.sidebar.title("## Choix des graphiques ou map que vous souhaitez afficher : ")

###################################################################

# PARTIE RAPH
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
        print(f"La combinaison indiquée n'existe pas : {rock_type} & {tectonic_setting}")


# prompt: pour un type d'éruption 'Type', représenter sur un graphique à double entrées (en ordonnée les valeurs de 'Dominant Rock Type', en abscisse les valeurs de 'Tectonic Setting') la probabilité d'occurence
# en considérant que la probabilité d'occurence est égale à la probabilité que ce binôme 'Dominant Rock Type' et 'Tectonic Setting' donne ce type d'éruption par rapport à tous les types d'éruption qu'il peut former

# demander le type d'éruption
eruption_type = st.selectbox("Quel type d'éruption ?", liste_type)
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
    print(f"Le type d'éruption '{eruption_type}' n'existe pas dans les données.")





#st.dataframe(df.style.highlight_max(axis=0)) surligne la valeur max de la colonne
#plot =df['alcohol']

#bot = st.checkbox("Afficher le graphique de la colonne alcohol : ")
#if bot :
#  st.write('Voici le graphique de la colonne alcohol : ')
#  st.line_chart(plot)
