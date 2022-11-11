import pandas as pd 




def load_CKD_data():
   # cvs_path = os.path.join(data_path, "CDK.cvs ")
   df =pd.read_csv(r'C:\Users\HP\Desktop\Final\EC449\Datasets\ckd.csv')
   df.head()
   return df


