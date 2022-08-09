# Import des librairies :
from math import pi
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# Import des fichiers, du modèle et du scaler :
path = "C:/Users/7700k/Desktop/Openclassrooms/Implémentez un modèle de scoring/API_streamlit/"
data = pd.read_csv(path+"data_api.csv", index_col=[0])
model = joblib.load("LGBM_model")
scaler = joblib.load("Scaler")

# Configuration du tableau de bord :
st.set_page_config(
    page_title="Moteur de recommandation crédit",
    layout="wide")

# Titre :
st.markdown("<h1 style='text-align: center; color: #5A5E6B;'>ANALYSE DU DOSSIER</h1>", unsafe_allow_html=True)

# Filtre :
id_filter = st.selectbox("Entrez identifiant client", pd.unique(data["SK_ID_CURR"]))
df = data[data["SK_ID_CURR"] == id_filter]

# Affichage de la prédiction :
prediction = int(model.predict(df.drop(columns=["TARGET","SK_ID_CURR"])))
proba = model.predict_proba(df.drop(columns=["TARGET","SK_ID_CURR"]))
proba = round(proba[0][0]*100,1)
if prediction == 0 :
	prob, pred = st.columns(2)
	with prob :
		st.success(f"Probabilités de remboursement : {proba} %")
	with pred :
		st.markdown("<h2 style='text-align: center; color: #44be6e;'>AVIS FAVORABLE</h2>", unsafe_allow_html=True)
else :
	prob, pred = st.columns(2)
	with prob :
		st.error(f"Probabilités de remboursement : {proba} %")
	with pred :
		st.markdown("<h2 style='text-align: center; color: #ff3d41;'>AVIS DÉFAVORABLE</h2>", unsafe_allow_html=True)
		# exemple de prêt refusé : ID=106854

# Ligne de démarcation :
st.markdown("***")

# Création d'une ligne de KPI :
age, revenus, montant_credit, duree_credit, montant_annuite = st.columns(5)

# Définition des KPI
age.metric(label="Age", value=f"{abs(int(round(df.DAYS_BIRTH/365,25)))} ans")
revenus.metric(label="revenus annuels", value=f"{int(round(df.AMT_INCOME_TOTAL))} $")
montant_credit.metric(label="Crédit demandé", value= f"{int(round(df.AMT_CREDIT))} $")
duree_credit.metric(label="Durée du crédit", value=f"{int(round(1/df.PAYMENT_RATE))} ans")
montant_annuite.metric(label="Montant des annuités", value=f"{int(round(df.AMT_ANNUITY))} $")

# Ligne de démarcation :
st.markdown("***")

# Tableau de scores :
all_features, score1, score2, score3 = st.columns(4)

with all_features :
	id_filter_features = st.selectbox("Toutes informations", pd.unique(df.columns))
	feature = df[id_filter_features]
	all_features.metric(label=id_filter_features, value=feature)

with score1 :
	if df["EXT_SOURCE_1"].isna().values :
		st.markdown("<h3 style='text-align: center; color: #ff3d41;'>Score 1 non renseigné</h3>", unsafe_allow_html=True)
	else :
		val = [(1-float(df["EXT_SOURCE_1"])), float(df["EXT_SOURCE_1"])]
		val.append(sum(val))
		colors = ["#ff3d41","#44be6e","white"]
		fig1, ax1 = plt.subplots()
		ax1.pie(val, colors=colors)
		ax1.add_artist(plt.Circle((0, 0), 0.6, color='white'))
		ax1.axis("equal")
		plt.title("Score 1")
		st.pyplot(fig1)

with score2 :
	if df["EXT_SOURCE_2"].isna().values :
		st.markdown("<h3 style='text-align: center; color: #ff3d41;'>Score 2 non renseigné</h3>", unsafe_allow_html=True)
	else :
		val = [(1-float(df["EXT_SOURCE_2"])), float(df["EXT_SOURCE_2"])]
		val.append(sum(val))
		colors = ["#ff3d41","#44be6e","white"]
		fig1, ax1 = plt.subplots()
		ax1.pie(val, colors=colors)
		ax1.add_artist(plt.Circle((0, 0), 0.6, color='white'))
		ax1.axis("equal")
		plt.title("Score 2")
		st.pyplot(fig1)

with score3 :
	if df["EXT_SOURCE_3"].isna().values :
		st.markdown("<h3 style='text-align: center; color: #ff3d41;'>Score 3 non renseigné</h3>", unsafe_allow_html=True)
	else :
		val = [(1-float(df["EXT_SOURCE_3"])), float(df["EXT_SOURCE_3"])]
		val.append(sum(val))
		colors = ["#ff3d41","#44be6e","white"]
		fig1, ax1 = plt.subplots()
		ax1.pie(val, colors=colors)
		ax1.add_artist(plt.Circle((0, 0), 0.6, color='white'))
		ax1.axis("equal")
		plt.title("Score 3")
		st.pyplot(fig1)

# Ligne de démarcation :
st.markdown("***")

# Graphiques et infos :
graphique1, graphique2 = st.columns(2, gap="large")

with graphique1 :
	# MinMax
	df_MM = df[["PAYMENT_RATE","AMT_ANNUITY","DAYS_EMPLOYED","INSTAL_DPD_MEAN","DAYS_BIRTH","AMT_GOODS_PRICE"]]
	df_MM = pd.DataFrame(scaler.transform(df_MM), columns=df_MM.columns)

	df_mean = data[["PAYMENT_RATE","AMT_ANNUITY","DAYS_EMPLOYED","INSTAL_DPD_MEAN","DAYS_BIRTH","AMT_GOODS_PRICE"]]
	df_mean = pd.DataFrame(scaler.transform(df_mean), columns=df_mean.columns)
	df_mean.mean()

	# set data :
	radar = pd.DataFrame({
    	"Groupe" : ["Client", "Moyenne défaut de paiement"],
    	"Rythme paiement" : [df_MM["PAYMENT_RATE"].fillna(0), df_mean["PAYMENT_RATE"].mean()],
    	"Montant des annuités" : [1 - df_MM["AMT_ANNUITY"].fillna(0), 1 - df_mean["AMT_ANNUITY"].mean()],
    	"Jours salariés (total)" : [1 - df_MM["DAYS_EMPLOYED"].fillna(0), 1 - df_mean["DAYS_EMPLOYED"].mean()],
   	"Fiabilité paiement" : [1 - df_MM["INSTAL_DPD_MEAN"].fillna(0), 1 - df_mean["INSTAL_DPD_MEAN"].mean()], 
    	"Jeunesse" : [df_MM["DAYS_BIRTH"].fillna(0), df_mean["DAYS_BIRTH"].mean()],  
    	"Montant des biens achetés" : [df_MM["AMT_GOODS_PRICE"].fillna(0), df_mean["AMT_GOODS_PRICE"].mean()]
	})

	# Initialisation de la figure :
	fig, ax = plt.subplots()

	# Nombre de variables :
	categories=list(radar)[1:]
	N = len(categories)

	# Nous allons tracer la première ligne du bloc de données.
	# Mais nous devons répéter la première valeur pour fermer le graphique circulaire :
	values=radar.loc[0].drop("Groupe").values.flatten().tolist()
	values += values[:1]

	# Quel sera l'angle de chaque axe ?
	angles = [n / float(N) * 2 * pi for n in range(N)]
	angles += angles[:1]

	# Initialisation du radar :
	ax = plt.subplot(111, polar=True)

	# Trace un axe par variable et ajoute les libélés :
	plt.xticks(angles[:-1], categories, color='grey', size=8)
	ax.tick_params(axis='x', which='major', pad=40)

	# Tracé
	ax.set_rlabel_position(0)
	plt.yticks([0.25,0.5,0.75], ["25%","50%","75%"], color="grey", size=7)
	plt.ylim(0,1)

	# Ind1
	values=radar.loc[0].drop("Groupe").values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, linewidth=1, linestyle='solid', label="Client")
	ax.fill(angles, values, 'b', alpha=0.1)
 
	# Ind2
	values=radar.loc[1].drop("Groupe").values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, linewidth=1, linestyle='solid', label="Moyenne")
	ax.fill(angles, values, 'r', alpha=0.1)
 
	# Add legend
	plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

	# Show the graph
	st.pyplot(fig)

with graphique2 :
	# Initialisation des valeurs et des paramètres :
	labels=["Revenu restant","Remboursement crédit"]
	values=[df["AMT_INCOME_TOTAL"].iloc[0] - df["AMT_ANNUITY"].iloc[0] , df["AMT_ANNUITY"].iloc[0]]
	colors = "#44be6e", "#ff3d41"
	explode = (0,0.2)

	# Graphique :
	fig1, ax1 = plt.subplots(figsize=(4,3))
	plt.title("Impact du crédit sur les revenus")
	ax1.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', textprops={"size":"smaller"})
	ax1.axis("equal")
	st.pyplot(fig1)