import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.datasets import load_wine

values, target = load_wine(return_X_y=True)
target_names = load_wine().target_names
feature_names = load_wine().feature_names

st.title("Les volcans de l'Holocène")

df = pd.DataFrame(values, columns=feature_names)
st.dataframe(df.style.highlight_max(axis=0))
plot =df['alcohol']

bot = st.checkbox("Afficher le graphique de la colonne alcohol : ")
if bot :
  st.write('Voici le graphique de la colonne alcohol : ')
  st.line_chart(plot)
