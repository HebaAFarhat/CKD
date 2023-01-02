from fastapi import FastAPI
import pandas as pd 
import uvicorn

import joblib


# [age, blood_pressure  , specific_gravity  , albumin , sugar, red_blood_cells, pus_cell,
#               pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium,
#               potassium, haemoglobin, packed_cell_volume, white_blood_cell_count , red_blood_cell_count,
#               hypertension, diabetes_mellitus, coronary_artery_disease, appetite, peda_edema,
#               aanemia


app = FastAPI(debug=True)

c = [48,80,1.02,1,0,0,"normal","notpresent","notpresent",121,36,1.2,0,0,15.4,44,7800,5.2,"yes","yes","no","good","no","no"]


@app.get('/')
def hello():
    return 'Chronic Kidney Disease'

@app.get("/predict")
def predict(age: int, blood_pressure: float  , spfecific_gravity:float  , albumin:float , sugar:float, red_blood_cells: str, pus_cell:str,
              pus_cell_clumps: str, bacteria:str, blood_glucose_random: float, blood_urea: float, serum_creatinine: float, sodium:float,
              potassium: float, haemoglobin:float, packed_cell_volume: float, white_blood_cell_count: float , red_blood_cell_count:float,
              hypertension:str, diabetes_mellitus:str, coronary_artery_disease:str, appetite:str, peda_edema:str,
              aanemia:str):
       data = {'age':age  , 'blood_pressure':blood_pressure, 'spfecific_gravity': spfecific_gravity, 'albumin':albumin, 'sugar':sugar, 'red_blood_cells': red_blood_cells, 'pus_cell':pus_cell,
              'pus_cell_clumps':pus_cell_clumps, 'bacteria':bacteria, 'blood_glucose_random':blood_glucose_random, 'blood_urea':blood_urea, 'serum_creatinine':serum_creatinine, 'sodium':sodium,
              'potassium': potassium, 'haemoglobin':haemoglobin, 'packed_cell_volume':packed_cell_volume, 'white_blood_cell_count':white_blood_cell_count, 'red_blood_cell_count':red_blood_cell_count,
              'hypertension':hypertension, 'diabetes_mellitus':diabetes_mellitus, 'coronary_artery_disease':coronary_artery_disease, 'appetite':appetite, 'peda_edema':peda_edema,
              'aanemia': aanemia}
       df = pd.DataFrame(data,index = [0])

      
       model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\model_jlib','rb'))
       makeprediction = model.predict(df)
       output = makeprediction
       return{"you have {}".format(output)}

if __name__ == "__main__":
    uvicorn.run(app)

    