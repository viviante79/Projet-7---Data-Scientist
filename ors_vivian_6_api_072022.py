# Import des librairies :
import pandas as pd
import streamlit as st
import joblib

# Import des fichiers, du modèle et du scaler :
data = pd.read_csv("dataset/data_api.csv", index_col=[0])
model = joblib.load("dataset/LGBM_model")
scaler = joblib.load("dataset/Scaler")

# Filtre :
id_filter = st.selectbox("Entrez identifiant client", pd.unique(data["SK_ID_CURR"]))
df = data[data["SK_ID_CURR"] == id_filter]

# Affichage de la prédiction :
query = df.drop(columns=["TARGET","SK_ID_CURR"])
score = list(model.predict_proba(query))
st.json({"Id" : int(df["SK_ID_CURR"]), "score" : score[0][0]})

query = data.drop(columns=["TARGET","SK_ID_CURR"])
score = list(model.predict_proba(query))
dictonnary = dict(zip(data["SK_ID_CURR"], score[:][:]))
st.write(dictionnary)
