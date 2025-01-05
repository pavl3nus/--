import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('report.csv', usecols=['MIP', 'STDIP', 'EKIP', 'SIP', 'MC', 'STDC', 'EKC', 'SC'])
data = MinMaxScaler().fit_transform(data)
avr = data.mean(axis = 0)
Star = [0.598,0.748,0.809,0.913,0.667,0.608,0.473,0.731]
print(avr[0])

from sklearn.linear_model import LogisticRegression

y = pd.read_csv('report.csv', usecols = ['TARGET']) #Отбираем данные для предсказаний
reg = LogisticRegression(random_state = 2019).fit(data, y.values.ravel()) #Обучаем модель
#Вводим свои параметры в скобочки и получаем предсказание отнесения нашей звезды
#Первое число - вероятность отнесения к не пульсару, второе - к пульсару
print(reg.predict_proba([Star]))

from sklearn.neighbors import KNeighborsClassifier
#n_neigbours - количество соседей, p = 1 - Манхэттенское расстояние, p = 2 - евклидово
#Кол-во соседей ставим равное кол-ву звезд в полученной ранее выборке
neigh = KNeighborsClassifier(n_neighbors=202, p=2)
neigh.fit(data, y.values.ravel())

print(neigh.kneighbors([Star])[0][0][0])
