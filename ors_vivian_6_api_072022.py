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
prediction = int(model.predict(df.drop(columns=["TARGET","SK_ID_CURR"])))
proba = model.predict_proba(df.drop(columns=["TARGET","SK_ID_CURR"]))
proba = round(proba[0][0]*100,1)
st.write(prediction, proba)