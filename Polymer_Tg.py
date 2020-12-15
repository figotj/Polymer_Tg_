import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from math import sqrt
import pickle

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Unique fingerprints found in dataset-1
unique_list = pickle.load(open("./Data/Fingerprint_unique_list.pkl","rb"))

#The total occurence of each substructure in dataset-1
Zero_Sum = pickle.load(open("./Data/Zero_Sum.pkl","rb"))

#The map between fingerprint hashcode and index
Corr_df = pickle.load(open("./Data/Corr_df.pkl","rb"))

#dataset-3 of 1 million polymers
df_1M = pd.read_csv("./Data/PI1M.csv")

#dataset-32 conjugated polymer input
df_32 = pd.read_csv("./Data/32_Conjugate_Polymer.txt", sep='\t')   
molecules = df_32.Smiles.apply(Chem.MolFromSmiles)
fps = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
fp = fps.apply(lambda m: m.GetNonzeroElements())

NumberOfZero = 6400 # optmized hyper-parameter to determine substructures used in model
MY_finger = []
for polymer in fp:
    my_finger = [0] * len(unique_list)
    for key in polymer.keys():
        if key in list(Corr_df[0]):
            index = Corr_df[Corr_df[0] == key]['index'].values[0]
            my_finger[index] = polymer[key]
    MY_finger.append(my_finger)
MY_finger_dataset_32 = pd.DataFrame(MY_finger)
X_dataset_32 = MY_finger_dataset_32[Zero_Sum[Zero_Sum < NumberOfZero].index]


#load Lasso_Fingerprint_model
Lasso_Fingerprint_model = pickle.load(open("./Model/LASSO.model","rb"))

#Prediction of Lasso_Fingerprint_model on the dataset-3 of 1 million polymers
X_1M = pickle.load(open("./Data/MY_finger_df_dataset3_{}.pkl".format(0),"rb"))
df_pred_dataset_3_lasso = list(Lasso_Fingerprint_model.predict((X_1M)))
for i in range(1,100):
     X_1M = pickle.load(open("./Data/MY_finger_df_dataset3_{}.pkl".format(i),"rb"))
     df = list(Lasso_Fingerprint_model.predict((X_1M)))
     df_pred_dataset_3_lasso = df_pred_dataset_3_lasso + df
     
#Performance of Lasso_Fingerprint_model on the dataset of 32 conjugated polymers
y_pred_dataset_32 = Lasso_Fingerprint_model.predict(X_dataset_32)
df_32['Tg_pred_Lasso_finger'] = y_pred_dataset_32
Xplot = df_32['Tg']
Yplot = df_32['Tg_pred_Lasso_finger']
final_R2 = r2_score(Xplot[:32],Yplot[:32])
final_MAE = mean_absolute_error(Xplot[:32],Yplot[:32])
rms = sqrt(mean_squared_error(Xplot[:32], Yplot[:32]))
MAPE = mean_absolute_percentage_error(Xplot[:32],Yplot[:32])
print('32 test R',final_R2)
print('32 test MAE',final_MAE)
print('32 test rms',rms)
print('32 test MAPE',MAPE)

#load DNN_Fingerprint_model
DNN_Fingerprint_model = load_model('./Model/DNN_Fingerprint_model')

#Prediction of DNN_Fingerprint_model on the dataset-3 of 1 million polymers
X_1M = pickle.load(open("./Data/MY_finger_df_dataset3_{}.pkl".format(0),"rb"))
df_pred_dataset_3_DNN = list(DNN_Fingerprint_model.predict((X_1M)))
for i in range(1,100):
     X_1M = pickle.load(open("./Data/MY_finger_df_dataset3_{}.pkl".format(i),"rb"))
     df = list(DNN_Fingerprint_model.predict((X_1M)))
     df_pred_dataset_3_DNN = df_pred_dataset_3_DNN + df
     
#Performance of DNN_Fingerprint_model on the dataset of 32 conjugated polymers
y_pred_dataset_32_DNN = DNN_Fingerprint_model.predict(X_dataset_32)
df_32['Tg_pred_DNN_finger'] = [arr[0] for arr in y_pred_dataset_32_DNN]
Xplot = df_32['Tg']
Yplot = df_32['Tg_pred_DNN_finger']
final_R2 = r2_score(Xplot[:32],Yplot[:32])
final_MAE = mean_absolute_error(Xplot[:32],Yplot[:32])
rms = sqrt(mean_squared_error(Xplot[:32], Yplot[:32]))
MAPE = mean_absolute_percentage_error(Xplot[:32],Yplot[:32])
print('32 test R',final_R2)
print('32 test MAE',final_MAE)
print('32 test rms',rms)
print('32 test MAPE',MAPE)

#dataset-3 of 1 million polymers with Tg predictions of Lasso_Fingerprint_model and DNN_Fingerprint_model
df_1M = pickle.load(open("./Data/df_1M_Tg.pkl","rb"))

