import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score
path = "/home/tehe/TNCY2A/projetep/Doc/DataPHM.xlsx"

df = pd.read_excel(path)
n = df.shape[0]
#nettoyage des données
M = []
j = 0 
for i in range(0,n):
    
    if int(df.iloc[i][9]) == 1:
        M.append(list(df.iloc[i]))
        j += 1

Sensor_GO = []
Sensor_CO = []
Sensor_CR = []
Sensor_P1 = []
Sensor_PW = []
Sensor_T3P = []
Sensor_T1 = []
Mode = []

for i in M:
    Mode.append(i[1])
    Sensor_GO.append(i[2])
    Sensor_CO.append(i[3])
    Sensor_CR.append(i[4])
    Sensor_P1.append(i[5])
    Sensor_PW.append(i[6])
    Sensor_T3P.append(i[7])
    Sensor_T1.append(i[8])

#on choisit après observation GO en fonction de Mode,CO et CR

#recherche période sans dégradation
Tpass = 0
while M[Tpass][0] == 0:
    Tpass += 1


#Apprentissage du modèle de régression
#On choisit une modèle de régression linéaire

features_train = []
features_test = []
lim = 70*Tpass//100
features_train.append(Mode[:lim+1])
features_train.append(Sensor_CO[:lim+1])
features_train.append(Sensor_CR[:lim+1])
features_train.append(Sensor_T1[:lim+1])
features_train.append(Sensor_GO[:lim+1])
features_train.append(Sensor_PW[:lim+1])
features_train.append(Sensor_P1[:lim+1])


features_train = np.array(features_train).T


features_test = np.array(features_test).T

scaler = StandardScaler()
Y = np.array(Sensor_T3P[:lim+1])

X = scaler.fit_transform(features_train)
