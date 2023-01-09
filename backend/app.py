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

  


ckd_counter = 0
notckd_counter = 0 


def cheak_for_ckd(classification):
     global ckd_counter
     global notckd_counter

     if(classification == ['ckd']):
          ckd_counter += 1
     else:
          notckd_counter += 1

     


@app.get('/')
def hello():
    return 'Chronic Kidney Disease'
data = {'age': 48, 'specific_gravity': 1.02, 'albumin':1,
                 'serum_creatinine':0, 
               'haemoglobin':44, 'packed_cell_volume':7800, 
              'hypertension':"yes", 'diabetes_mellitus':"yes", 'peda_edema':"no"}
@app.post("/predict")
def predict(age: int,   specific_gravity:float  , albumin:float , serum_creatinine: float,  haemoglobin:float, packed_cell_volume: float , 


              hypertension:str, diabetes_mellitus:str,peda_edema:str):
       
      data = {'age':age  , 'specific_gravity': specific_gravity, 'albumin':albumin,  'serum_creatinine':serum_creatinine, 
               'haemoglobin':haemoglobin, 'packed_cell_volume':packed_cell_volume, 'hypertension':hypertension, 'diabetes_mellitus':diabetes_mellitus,'peda_edema':peda_edema,
              }
      df = pd.DataFrame(data,index = [0])



       # prediction of ET classifier
      model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\decision_tree_jlib','rb'))
      dt_prediction = model.predict(df)
      output = dt_prediction
      print(output)


      cheak_for_ckd(output)
      print(ckd_counter)

    
     
       # prediction of ET classifier
      model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\extra_tree_model_jlib','rb'))
      ET_prediction = model.predict(df)
      output = ET_prediction
      print(output)

      cheak_for_ckd(output)
      print(ckd_counter)





    #    # prediction of GB classifier
    #   model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\gradient_boosting_model_jlib','rb'))
    #   GB_prediction = model.predict(df)
    #   output = GB_prediction
    #   cheak_for_ckd(output)

       # prediction of SVC classifier
      model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\svc_model_jlib','rb'))
      SVC_prediction = model.predict(df)
      output = SVC_prediction
      print(output)

      cheak_for_ckd(output)
      print(ckd_counter)


       # prediction of RF classifier
      model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\random_forest_model_jlib','rb'))
      RF_prediction = model.predict(df)
      output = RF_prediction
      print(output)

      cheak_for_ckd(output)
      print(ckd_counter)


    #    # prediction of SGB classifier
    #   model = joblib.load(open('C:\\Users\\Hiba\\Desktop\\EC449\\Code\\stochastic_gradient_Boosting_model_jlib','rb'))
    #   SGB_prediction = model.predict(df)
    #   output = SGB_prediction
    #   print(output)

      if (ckd_counter > notckd_counter):
           return "This pataint may have CKD, please follow up with more screeing, Thank You."
      else:
           return "This paitiant doesn't have CKD, Stay healthy."


if __name__ == "__main__":
    uvicorn.run(app)
    