import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

# Import des fichiers, du modèle et du scaler :
data = pd.read_csv("dataset/data_api.csv", index_col=[0])
model = joblib.load("dataset/LGBM_model")

# Create the app object
app = FastAPI()

@app.get('/all')
def get_all() :
    query = data.drop(columns=["TARGET","SK_ID_CURR"])
    score = list(model.predict_proba(query))

    dictionnary = dict(zip(data["SK_ID_CURR"], score[:][:]))
    return jsonable_encoder(str(dictionnary))

#  Route with a single parameter, returns the parameter within a message
#  Located at: http://127.0.0.1:5000/AnyNameHere
@app.get('/{ID}')
def get_ID(ID : int) :
    query = data[data["SK_ID_CURR"]==ID].drop(columns=["TARGET","SK_ID_CURR"])
    score = list(model.predict_proba(query))

    return {"ID" : ID, "score" : score[0][0]}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:5000

if __name__ == '__main__':
    uvicorn.run(app, host=https://viviante79-projet-7-data-scienti-ors-vivian-5-api-072022-ucevpb.streamlitapp.com/)
