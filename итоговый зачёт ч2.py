import numpy as np
from keras.api.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

dim = 784 # 28*28 
X_train = X_train.reshape(len(X_train), dim)

ev_ = 0.84 # доля объясненной дисперсии должна превышать это значение
pca = PCA(svd_solver='full')
pca.fit(X_train)
explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),3)

M = np.argmax(explained_variance > ev_) + 1

print(f'Необходимо {M} компонент(ы)')

pca = PCA(n_components = M, svd_solver = 'full')
pca.fit(X_train)
X_train_transformed = pca.transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(X_train_transformed, y_train, test_size = 0.3, random_state =95)
print(f'Выборочное среднее: {sum([i[0] for i in X_train]) / len(X_train)}')

rfc = RandomForestClassifier(criterion='gini', min_samples_leaf=10, max_depth = 20, n_estimators = 10, random_state = 95)
clf = OneVsRestClassifier(rfc).fit(X_train, y_train)
y_pred = clf.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
P = 4
print(f'{P}: {CM[P][P]}')

reg = LogisticRegression(solver='lbfgs', random_state = 95)
clf = OneVsRestClassifier(reg).fit(X_train, y_train)
y_pred = clf.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
P = 6
print(f'{P}: {CM[P][P]}')

reg = DecisionTreeClassifier(criterion='gini', min_samples_leaf=10, max_depth=20, random_state=95)
clf = OneVsRestClassifier(reg).fit(X_train, y_train)
y_pred = clf.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
P = 7
print(f'{P}: {CM[P][P]}')
