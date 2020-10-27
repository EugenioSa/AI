import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix


data = pd.read_csv("winequalityred.csv") 
data['quality'].dtype
data['quality_label'] = (data['quality'] > 6.5)*1

y = data["quality_label"]
x = data[["volatile acidity","citric acid","total sulfur dioxide","density","sulphates","alcohol"]]
X = x.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 42)

  
# Creamos un objeto arbol
tree = DecisionTreeClassifier(max_depth=10, random_state = 0, max_leaf_nodes=10, criterion="gini")
tree.fit(X_train, y_train)

np.random.seed() 
idxs = np.random.randint(X.shape[0], size=10)
instancias = X[idxs,:]

y_pred = tree.predict(instancias)
print("Predicciones muestra: ")
for i, idx in enumerate(idxs):
    print(f'Instancia {idx}. Etiqueta real: {y[idx]}. Etiqueta predicha: {y_pred[i]}')



# Predecimos sobre nuestro set de entrenamieto
y_pred = tree.predict(X_test)

#Obtener exactitud de modelo
print("Accuracy score: " + str(accuracy_score(y_pred,y_test)))


plot_confusion_matrix(tree, X_test, y_test, cmap=plt.cm.Blues, values_format = '.0f')
plt.show()
