import numpy as np
import pandas as pd

dic = {"Nombres":["Maria","Carlos","juan","Alejandra","Gustavo","Diana","Adriana","German","James","Libia","Rafael","Oliverio","Gloria", "Mario"],
       "Apellidos":["Gomez","Velez","Porras","Perea","Perez","Castro","Molina","Penagos","Rodriguez","Perdomo","Cook","Serrato","Castaño","Vergara"],
       "Edades":[23,15,45,34,67,18,40,26,30,60,10,55,20,15],
       "Salarios":[20000000,1000000,5000000,10000000,2500000,3000000,5000000,3000000,30000000,10000000,50000,10000000,800000,500000],
       "Sexo":["Femenino","Masculino","Masculino","Femenino","Masculino","Femenino","Femenino","Masculino","Masculino","Femenino","Masculino","Masculino",
               "Femenino","Masculino"],
       "Gastos_Mensuales":[5000000,300000,1000000,600000,500000,300000,100000,800000,5000000,600000,10000,2000000,200000,100000]}

data = pd.DataFrame(dic)
print(data)

X = data["Salarios"].values
X = X.reshape(len(X), 1)
y = data["Gastos_Mensuales"].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

from sklearn.svm import SVR

svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)

svr.fit(X, y)
y_pred = svr.predict(X_test)
print(y_pred)
