import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import random



class LogisticRegressionS:
    
    #Sigmoid function
    def hyphotesis(self, X, thetha):
        z = np.dot(thetha, X.T)
        return 1/(1 + np.exp((-1) * (z))) - 0.0000001

    #Cost function
    def costFunction(self, X, y, thetha):
        yp = self.hyphotesis(X, thetha)
        return -(1/len(X)) * np.sum(y*np.log(yp) + (1-y)*np.log(1-yp))

    #Gradient Descent
    def gradientDescent(self, X, y, thetha, alpha, epochs):
        m = len(X)
        J = [self.costFunction(X, y, thetha)]
        for i in range(0, epochs):
            h = self.hyphotesis(X, thetha)
            for i in range(0, len(X.columns)):
                thetha[i] -= (alpha/m) * np.sum((h-y)*X.iloc[:, i])
            J.append(self.costFunction(X, y, thetha))
        return J, thetha

    def predict(self, X, y, thetha, alpha, epochs):
        J, th = self.gradientDescent(X, y, thetha, alpha, epochs)
        h = self.hyphotesis(X, thetha)
        for i in range(len(h)):
            h[i] = 1 if h[i] >= 0.5 else 0
        y = list(y)
        acc = np.sum([y[i] == h[i] for i in range(len(y))]) / len(y)
        #print(y, h)
        return J, acc, h

    def predict2(self, X, y, thetha, alpha, epochs):
        J, th = self.gradientDescent(X, y, thetha, alpha, epochs)
        h = self.hyphotesis(X, thetha)
        for i in range(len(h)):
            h[i] = 1 if h[i] >= 0.5 else 0
        
        #print(y, h)
        return J, h





      

data = pd.read_csv("data_placement.csv")
data['status'] = (data['status'] == "Placed")*1
data['workex'] = (data['workex'] == "Yes")*1
data['ssc_b'] = (data['ssc_b'] == "Central")*1
data['hsc_b'] = (data['hsc_b'] == "Central")*1
data['gender'] = (data['gender'] == "F")*1
data['specialisation'] = (data['specialisation'] == "Mkt&Fin")*1
df = pd.DataFrame(data)
df['salary'] = df['salary'].fillna(0)

y = data["status"]
X = data[["gender","ssc_p","hsc_p","degree_p", "workex","etest_p","specialisation"]]
X2 = X.values
y2 = y.values
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 1/5, random_state = 0)

clf = LogisticRegression(random_state=0, max_iter=2000).fit(X_train, y_train)

idxs = np.random.randint(X2.shape[0], size=10)
instancias = X2[idxs,:]

'''y_pred = clf.predict(instancias)
print("Train")
for i, idx in enumerate(idxs):
    print(f'Instancia {idx}. Etiqueta real: {y[idx]}. Etiqueta predicha: {y_pred[i]}')

print("Mean accuracy: ", clf.score(X_train, y_train))'''

y_pred = clf.predict(X_test)
print("Test")
for i in range(len(X_test)):
    print(f'Instancia {i}. Etiqueta real: {y_test[i]}. Etiqueta predicha: {y_pred[i]}')

#print(clf.predict_proba(X_test))
print("Mean accuracy Framework: ", clf.score(X, y))


logR = LogisticRegressionS()
thetha = [0.5]*len(X.columns)
lr = 0.0001
iterations = 2000
#print(y)
J, acc, h = logR.predict(X, y, thetha, lr, iterations)
print("Mean accuracy By Hand: ", acc)
plt.figure(figsize = (12, 8))
plt.scatter(range(0, len(J)), J)
#plt.show()

answers = []
yH = [0]

answers.append(int(input("Your gender M(0) F(1): ")))

answers.append(int(input("Secondary Education Percentage: ")))

answers.append(int(input("Higher Secondary Education Percentage: ")))

answers.append(int(input("Degree Percentage: ")))

answers.append(int(input("Work Experience Yes(1) No(0): ")))

answers.append(int(input("Employability Test Percentage: ")))

answers.append(int(input("Specialisation Mkt&Fin(1) Mkt&HR(0): ")))

arr = []
arr.append(answers)
dataframe=pd.DataFrame(arr, columns=[["gender","ssc_p","hsc_p","degree_p", "workex","etest_p","specialisation"]]) 
line = pd.DataFrame({"gender": answers[0], "ssc_p": answers[1], "hsc_p": answers[2], "degree_p": answers[3], "workex":answers[4], "etest_p": answers[5], "specialisation": answers[6] }, index=[215])
X = X.append(line, ignore_index=False)
X = X.sort_index().reset_index(drop=True)

#X.loc[-1] = answers
#X.index = X.index + 1  # shifting index
#X = X.sort_index()  # sorting by index

y.loc[-1] = int(yH[0])
y.index = y.index + 1  # shifting index
y = y.sort_index()  # sorting by index

#line = pd.DataFrame({"status": yH[0] }, index=[215])
#y = y.append(line, ignore_index=False)
#y = y.sort_index().reset_index(drop=True)





y_pred = clf.predict(dataframe)
print("predicted by Framework: ", y_pred)


J, h = logR.predict2(X,y, thetha, lr, iterations)
print("predicted by Hand: ", h[215])




