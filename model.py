import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from IPython.display import display

names = ['test_date', 'cough', 'fever', 'sore_throat', 'shortness_of_breath','head_ache','age_60_and_above' , 'gender' , 'test_indication' , 'corona_result']

# Read dataset to pandas dataframe
dataset = pd.read_csv('data/corona.csv', names=names , low_memory=False)

dataset.head()

dataset.drop(dataset.index[dataset['corona_result'] == 'other'], inplace = True)

dataset = dataset.replace(to_replace='None', value=np.nan).dropna()

dataset['test_date'] = pd.factorize(dataset['test_date'])[0]
dataset['head_ache'] = pd.factorize(dataset['head_ache'])[0]
dataset['age_60_and_above'] = pd.factorize(dataset['age_60_and_above'])[0]
dataset['gender'] = pd.factorize(dataset['gender'])[0]
dataset['test_indication'] = pd.factorize(dataset['test_indication'])[0]
dataset['corona_result'] = pd.factorize(dataset['corona_result'])[0]

dataset["test_date"] = pd.to_numeric(dataset["test_date"], downcast="float")
dataset["cough"] = pd.to_numeric(dataset["cough"], downcast="float")
dataset["fever"] = pd.to_numeric(dataset["fever"], downcast="float")
dataset["sore_throat"] = pd.to_numeric(dataset["sore_throat"], downcast="float")
dataset["shortness_of_breath"] = pd.to_numeric(dataset["shortness_of_breath"], downcast="float")
dataset["head_ache"] = pd.to_numeric(dataset["head_ache"], downcast="float")
dataset["age_60_and_above"] = pd.to_numeric(dataset["age_60_and_above"], downcast="float")
dataset["gender"] = pd.to_numeric(dataset["gender"], downcast="float")
dataset["test_indication"] = pd.to_numeric(dataset["test_indication"], downcast="float")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 9].values
display(pd.DataFrame(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#save model in output directory
joblib.dump(classifier,'output/covid-predictor_model.pkl')