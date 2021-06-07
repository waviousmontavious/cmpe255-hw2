#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#%%
with open('data/train.dat') as file:
    train_data = np.loadtxt(file, delimiter=' ')
with open('data/train.labels') as file:
    train_labels = np.loadtxt(file)

train_labels = train_labels.astype(int)

#%%
pca = PCA(n_components=30).fit(train_data)
print(f'Explained Variance Ratios: {pca.explained_variance_ratio_}')
print(f'Total Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%')

X = pca.fit_transform(train_data)
X_train, X_test, y_train, y_test = train_test_split(X, train_labels, test_size=0.20, random_state=40)

#%%
knn = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance',
    algorithm='brute',
    leaf_size=50,
    metric='minkowski',
    p=1
)

pipe = Pipeline(steps=[
    #('scaler', StandardScaler()),
    ('knn', knn),
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(f'accuracy score = {score:.4f}')

#%%
with open('data/test.dat') as file:
    test_data = np.loadtxt(file, delimiter=' ')

#%%
test_data = pca.fit_transform(test_data)

test_labels = pipe.predict(test_data)

#%%
np.savetxt(f'predictions_f1_{score:.4f}.txt', test_labels, '%i')
# %%
print(set(train_labels))
# %%
